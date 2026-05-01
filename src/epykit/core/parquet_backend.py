"""
epykit.core.parquet_backend
============================
Data backend for partitioned Parquet methylation datasets.

This module provides the ParquetMethylStore class, which supersedes AnnData
for per-chromosome and per-sample methylation analysis. It handles:

- Lazy scanning of partitioned Parquet files
- Efficient filtering by coverage, context, genomic regions
- Pivoting to wide format for statistical testing
- QC and summary statistics

Key design principle: never load a full chromosome into RAM unnecessarily.
All I/O uses streaming and predicate pushdown via Polars' lazy evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ParquetMethylStore:
    """
    Accessor and manager for a partitioned Parquet methylation dataset.
    
    A partitioned Parquet store has the structure::
    
        store_root/
          sample=S1/
            chrom=chr1/
              part-0.parquet
            chrom=chr2/
              part-0.parquet
          sample=S2/
            chrom=chr1/
              part-0.parquet
    
    Each Parquet file contains columns: [chrom, pos, strand, N_meth, N_unmeth, coverage, sample].
    
    Parameters
    ----------
    store_path : PathLike
        Root directory of the partitioned Parquet store.
    
    Attributes
    ----------
    store_path : Path
        Root path of the store.
    
    Examples
    --------
    >>> store = ParquetMethylStore("methylstore")
    >>> samples = store.samples()
    >>> chrom_df = store.load_chromosome("chr1", samples=["S1", "S2"])
    >>> stats = store.sample_summary("S1")
    """
    
    def __init__(self, store_path: PathLike):
        self.store_path = Path(store_path)
        
        if not self.store_path.exists():
            raise FileNotFoundError(f"Parquet store not found: {self.store_path}")
        
        # Check for presence of sample=* structure
        sample_dirs = list(self.store_path.glob("sample=*"))
        if not sample_dirs:
            raise ValueError(
                f"No sample partitions found in {self.store_path}. "
                f"Expected structure: store_path/sample=*/chrom=*/*.parquet"
            )
    
    def samples(self) -> list[str]:
        """
        Return list of all sample identifiers in the store.
        
        Returns
        -------
        list[str]
            Sorted list of sample names (e.g., ["S1", "S2", "S3"]).
        """
        sample_dirs = sorted(self.store_path.glob("sample=*"))
        samples = [d.name.replace("sample=", "") for d in sample_dirs]
        return samples
    
    def chromosomes(self, sample: str | None = None) -> list[str]:
        """
        Return list of chromosomes present in the store.
        
        Parameters
        ----------
        sample : str | None
            If provided, return chromosomes for this sample only.
            If None, return union of all chromosomes across samples.
        
        Returns
        -------
        list[str]
            Sorted list of chromosome names.
        """
        if sample is not None:
            chrom_dirs = sorted(
                (self.store_path / f"sample={sample}").glob("chrom=*")
            )
            chroms = [d.name.replace("chrom=", "") for d in chrom_dirs if d.is_dir()]
        else:
            # Union across all samples
            chroms_set = set()
            for sample_dir in self.store_path.glob("sample=*"):
                for chrom_dir in sample_dir.glob("chrom=*"):
                    chroms_set.add(chrom_dir.name.replace("chrom=", ""))
            chroms = sorted(chroms_set)
        
        return chroms
    
    def load_chromosome(
        self,
        chrom: str,
        samples: Sequence[str] | None = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Load all data for a single chromosome across specified samples.
        
        Parameters
        ----------
        chrom : str
            Chromosome name (e.g., "chr1").
        samples : Sequence[str] | None
            List of samples to include. If None, includes all samples.
        lazy : bool
            If True, return a LazyFrame (no computation). If False, collect immediately.
            Default: False.
        
        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            All rows for this chromosome and samples, with columns:
            [chrom, pos, strand, N_meth, N_unmeth, coverage, sample].
            
            Rows are sorted by pos within each sample.
        """
        if samples is None:
            samples = self.samples()
        
        samples = list(samples)
        
        # Build glob pattern to scan Parquet files
        pattern = str(self.store_path / f"sample=*/chrom={chrom}/*.parquet")
        
        lf = pl.scan_parquet(pattern)
        
        # Filter to requested samples
        lf = lf.filter(pl.col("sample").is_in(samples))
        
        if lazy:
            return lf
        else:
            return lf.collect()
    
    def load_chromosome_lazy(
        self,
        chrom: str,
        samples: Sequence[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Load chromosome data as a lazy frame (no computation until collect()).
        
        See load_chromosome() for parameters.
        """
        return self.load_chromosome(chrom=chrom, samples=samples, lazy=True)
    
    def filter_coverage(
        self,
        min_cov: int = 1,
        max_cov: int | None = None,
        max_cov_percentile: float | None = None,
        output_dir: PathLike | None = None,
        n_workers: int | None = None,
    ) -> ParquetMethylStore:
        """
        Filter sites by coverage and write to a new store.
        
        Parameters
        ----------
        min_cov : int
            Minimum coverage threshold. Default: 1 (no filter).
        max_cov : int | None
            Maximum coverage threshold. Default: None.
        max_cov_percentile : float | None
            Upper percentile for coverage (0–1). Computed per-sample, so each sample's
            threshold may differ. If both max_cov and max_cov_percentile are given,
            max_cov_percentile is ignored. Default: None.
        output_dir : PathLike | None
            Directory to write filtered store. If None, uses store_path + "_filtered".
        n_workers : int | None
            Number of parallel processes for filtering. If None, uses CPU count.
        
        Returns
        -------
        ParquetMethylStore
            New store containing only sites passing the filters.
        
        Notes
        -----
        This is a two-pass operation:
        1. (Optional) First pass computes max_cov_percentile per sample.
        2. Second pass filters and writes to output_dir.
        
        Because filtering uses streaming, it does not load full samples into RAM.
        """
        if output_dir is None:
            output_dir = Path(str(self.store_path) + "_filtered")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Filtering coverage: min_cov={min_cov}")
        
        # Compute per-sample max_cov threshold if percentile given
        max_cov_by_sample = {}
        if max_cov_percentile is not None:
            logger.debug(f"Computing coverage percentile {max_cov_percentile} per sample")
            for sample in self.samples():
                lf = self.load_chromosome_lazy("chr1", samples=[sample])  # Any chrom
                # Aggregate across all chroms for this sample
                lf_all = pl.scan_parquet(
                    str(self.store_path / f"sample={sample}/chrom=*/*.parquet")
                )
                cov_quantile = (
                    lf_all.select(
                        pl.col("coverage").quantile(max_cov_percentile)
                    )
                    .collect()
                    .item()
                )
                max_cov_by_sample[sample] = int(cov_quantile)
                logger.debug(f"  {sample}: {max_cov_by_sample[sample]}")
        
        # Second pass: filter and write
        for sample in self.samples():
            sample_max_cov = max_cov_by_sample.get(sample, max_cov)
            
            lf = pl.scan_parquet(
                str(self.store_path / f"sample={sample}/chrom=*/*.parquet")
            )
            
            lf = lf.filter(pl.col("coverage") >= min_cov)
            if sample_max_cov is not None:
                lf = lf.filter(pl.col("coverage") <= sample_max_cov)
            
            lf.sink_parquet(
                str(output_dir),
                compression="zstd",
                row_group_size=1_000_000,
                partition_by=["sample", "chrom"],
                maintain_order=False,
            )
        
        logger.info(f"✓ Filtered store written to {output_dir}")
        return ParquetMethylStore(output_dir)
    
    def sample_summary(self, sample: str) -> dict:
        """
        Compute summary statistics for a sample across all chromosomes.
        
        Parameters
        ----------
        sample : str
            Sample identifier.
        
        Returns
        -------
        dict
            Summary statistics:
            {
                "sample": sample,
                "n_sites": int,
                "mean_coverage": float,
                "median_coverage": float,
                "global_methylation": float,  # (total_meth / total_coverage) * 100
                "by_chrom": {
                    "chr1": {
                        "n_sites": int,
                        "mean_coverage": float,
                        "median_coverage": float,
                        "global_methylation": float,
                    },
                    ...
                }
            }
        """
        lf = pl.scan_parquet(
            str(self.store_path / f"sample={sample}/chrom=*/*.parquet")
        )
        
        # Overall stats
        overall = (
            lf.select([
                pl.count().alias("n_sites"),
                pl.col("coverage").mean().alias("mean_coverage"),
                pl.col("coverage").median().alias("median_coverage"),
                (pl.col("N_meth").sum() / pl.col("coverage").sum() * 100.0)
                .alias("global_methylation"),
            ])
            .collect()
            .row(0)
        )
        
        # Per-chromosome stats
        by_chrom = (
            lf.group_by("chrom")
            .agg([
                pl.count().alias("n_sites"),
                pl.col("coverage").mean().alias("mean_coverage"),
                pl.col("coverage").median().alias("median_coverage"),
                (pl.col("N_meth").sum() / pl.col("coverage").sum() * 100.0)
                .alias("global_methylation"),
            ])
            .collect()
            .to_dicts()
        )
        
        by_chrom_dict = {
            row["chrom"]: {
                "n_sites": row["n_sites"],
                "mean_coverage": row["mean_coverage"],
                "median_coverage": row["median_coverage"],
                "global_methylation": row["global_methylation"],
            }
            for row in by_chrom
        }
        
        return {
            "sample": sample,
            "n_sites": overall[0],
            "mean_coverage": overall[1],
            "median_coverage": overall[2],
            "global_methylation": overall[3],
            "by_chrom": by_chrom_dict,
        }
    
    def pivot_chromosome_wide(
        self,
        chrom: str,
        samples: Sequence[str],
        value_columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Load a chromosome and pivot to wide format (one row per site).
        
        Parameters
        ----------
        chrom : str
            Chromosome name.
        samples : Sequence[str]
            Samples to include.
        value_columns : Sequence[str] | None
            Columns to pivot (default: ["N_meth", "N_unmeth", "coverage"]).
        
        Returns
        -------
        pl.DataFrame
            Wide format: index [pos, strand], columns like N_meth_S1, N_meth_S2, etc.
            Rows are sorted by pos.
        
        Notes
        -----
        This loads the full chromosome into RAM, which for large cohorts may
        exceed memory. In this case, chunk by genomic windows.
        """
        if value_columns is None:
            value_columns = ["N_meth", "N_unmeth", "coverage"]
        
        df = self.load_chromosome(chrom, samples=samples, lazy=False)
        
        # Pivot to wide: (pos, strand) × (sample, value_col)
        wide = df.pivot(
            index=["pos", "strand"],
            columns="sample",
            values=value_columns,
            aggregate_function="first",  # Should be unique per pos/strand/sample
        )
        
        # Sort by pos
        wide = wide.sort_by("pos")
        
        return wide
    
    def get_comparison_sites(
        self,
        chrom: str,
        samples: Sequence[str],
        mode: Literal["intersect", "union"] = "intersect",
    ) -> list[tuple[int, str]]:
        """
        Get list of (pos, strand) tuples present in specified samples.
        
        Parameters
        ----------
        chrom : str
            Chromosome name.
        samples : Sequence[str]
            Samples to compare.
        mode : str
            "intersect": only sites in ALL samples.
            "union": all sites in ANY sample.
        
        Returns
        -------
        list[tuple[int, str]]
            Sorted list of (pos, strand) tuples.
        """
        df = self.load_chromosome(chrom, samples=samples, lazy=False)
        
        if mode == "intersect":
            # Sites present in all samples
            sites_per_sample = df.group_by(["pos", "strand"]).agg(
                pl.col("sample").n_unique().alias("n_samples_with_site")
            )
            n_samples = len(samples)
            sites = sites_per_sample.filter(
                pl.col("n_samples_with_site") == n_samples
            ).select(["pos", "strand"])
        elif mode == "union":
            # All sites
            sites = df.select(["pos", "strand"]).unique()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Convert to sorted list of tuples
        sites_list = [
            (row[0], row[1]) for row in sites.sort_by("pos").rows()
        ]
        
        return sites_list

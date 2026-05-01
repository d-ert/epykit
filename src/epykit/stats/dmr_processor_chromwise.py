"""Chromosome-wise differential methylation processing."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from epykit.core.parquet_backend import ParquetMethylStore
from epykit.stats.tests import (
    _apply_fdr,
    _fisher_exact_from_arrays,
    _glm_lrt_from_arrays,
    _limma_ebayes_from_arrays,
    _mean_diff_from_count_arrays,
)

logger = logging.getLogger(__name__)


def _as_sample_index(metadata: pd.DataFrame, samples: Sequence[str]) -> pd.DataFrame:
    if set(samples).issubset(set(metadata.index.astype(str))):
        return metadata.loc[list(samples)].copy()

    if "sample_id" in metadata.columns:
        indexed = metadata.set_index("sample_id", drop=False)
        if set(samples).issubset(set(indexed.index.astype(str))):
            return indexed.loc[list(samples)].copy()

    raise ValueError(
        "metadata must be indexed by sample name or contain a 'sample_id' column"
    )


def _sanitize_group_label(label: str) -> str:
    return str(label).replace(" ", "_")


def _build_design_matrices(
    metadata: pd.DataFrame,
    *,
    group_col: str,
    group_b: str,
    covariates: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    design_full = pd.DataFrame(index=metadata.index)
    design_full["intercept"] = 1.0
    design_full["treatment"] = (
        metadata[group_col].astype(str) == str(group_b)
    ).astype(float)

    design_reduced = pd.DataFrame(index=metadata.index)
    design_reduced["intercept"] = 1.0

    if covariates:
        for cov_col in covariates:
            if cov_col not in metadata.columns:
                continue
            values = metadata[cov_col]
            if values.dtype.kind in ("O", "U", "S") or str(values.dtype) == "category":
                dummies = pd.get_dummies(values.astype(str), prefix=cov_col, drop_first=True)
                if not dummies.empty:
                    design_full = pd.concat([design_full, dummies], axis=1)
                    design_reduced = pd.concat([design_reduced, dummies], axis=1)
            else:
                numeric = values.astype(float)
                design_full[cov_col] = numeric
                design_reduced[cov_col] = numeric

    return design_full.to_numpy(dtype=float), design_reduced.to_numpy(dtype=float)


def _extract_pivot_arrays(
    df: pl.DataFrame,
    samples: Sequence[str],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    index_cols = ["chrom", "pos", "strand"]

    meth_wide = df.pivot(
        index=index_cols,
        columns="sample",
        values="N_meth",
        aggregate_function="first",
    )
    cov_wide = df.pivot(
        index=index_cols,
        columns="sample",
        values="coverage",
        aggregate_function="first",
    )

    meth_wide = meth_wide.rename(
        {sample: f"meth_{sample}" for sample in samples if sample in meth_wide.columns}
    )
    cov_wide = cov_wide.rename(
        {sample: f"cov_{sample}" for sample in samples if sample in cov_wide.columns}
    )

    wide = meth_wide.join(cov_wide, on=index_cols, how="inner").drop_nulls()
    wide = wide.sort(["pos", "strand"])

    meth_cols = [f"meth_{sample}" for sample in samples]
    cov_cols = [f"cov_{sample}" for sample in samples]

    meth_counts = wide.select(meth_cols).to_numpy()
    cov_counts = wide.select(cov_cols).to_numpy()
    return wide, meth_counts, cov_counts


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "chr": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "strand": pl.Utf8,
            "pvalue": pl.Float64,
            "mean_diff": pl.Float64,
        }
    )


def process_chromosome(
    methylstore: ParquetMethylStore,
    chrom: str,
    samples: Sequence[str],
    group_col: str,
    metadata: pd.DataFrame,
    *,
    test: str = "auto",
    covariates: Sequence[str] | None = None,
    overdispersion: bool = True,
    limma_chunk_size: int = 100000,
    limma_alpha: float = 0.5,
) -> pl.DataFrame:
    """Run differential methylation testing for a single chromosome."""
    ordered_samples = list(samples)
    meta = _as_sample_index(metadata, ordered_samples)

    groups = meta[group_col].dropna().astype(str).unique().tolist()
    if len(groups) != 2:
        raise ValueError(f"group_col must have exactly 2 groups, found: {groups}")

    group_a, group_b = sorted(groups)
    group_a_samples = meta.index[meta[group_col].astype(str) == group_a].tolist()
    group_b_samples = meta.index[meta[group_col].astype(str) == group_b].tolist()
    if not group_a_samples or not group_b_samples:
        raise ValueError("Both groups must contain at least one sample")

    effective_test = test
    if test == "auto":
        effective_test = "fisher" if (len(group_a_samples) == 1 and len(group_b_samples) == 1) else "limma"

    logger.info(
        "process_chromosome(%s): %d samples (%s=%d, %s=%d), test='%s'",
        chrom,
        len(ordered_samples),
        group_a,
        len(group_a_samples),
        group_b,
        len(group_b_samples),
        effective_test,
    )

    df = methylstore.load_chromosome(chrom, samples=ordered_samples, lazy=False)
    if df.is_empty():
        return _empty_result()

    site_counts = (
        df.group_by(["chrom", "pos", "strand"])
        .agg(pl.col("sample").n_unique().alias("n_samples"))
    )
    df = df.join(
        site_counts.filter(pl.col("n_samples") == len(ordered_samples)).select(["chrom", "pos", "strand"]),
        on=["chrom", "pos", "strand"],
        how="inner",
    )

    if df.is_empty():
        return _empty_result()

    wide, meth_counts, cov_counts = _extract_pivot_arrays(df, ordered_samples)
    if wide.is_empty():
        return _empty_result()

    group_a_idx = [ordered_samples.index(sample) for sample in group_a_samples]
    group_b_idx = [ordered_samples.index(sample) for sample in group_b_samples]

    meth_a = meth_counts[:, group_a_idx]
    cov_a = cov_counts[:, group_a_idx]
    meth_b = meth_counts[:, group_b_idx]
    cov_b = cov_counts[:, group_b_idx]

    mean_diff, mean_a, mean_b = _mean_diff_from_count_arrays(meth_a, cov_a, meth_b, cov_b)

    if effective_test == "fisher":
        pvalues, log_ors, _ = _fisher_exact_from_arrays(meth_a, cov_a, meth_b, cov_b)
        result = pl.DataFrame(
            {
                "chr": wide.get_column("chrom"),
                "start": (wide.get_column("pos") - 1).cast(pl.Int64),
                "end": wide.get_column("pos").cast(pl.Int64),
                "strand": wide.get_column("strand"),
                "pvalue": pvalues,
                "mean_diff": mean_diff,
                f"mean_meth_{_sanitize_group_label(group_a)}": mean_a,
                f"mean_meth_{_sanitize_group_label(group_b)}": mean_b,
                "log2_odds_ratio": log_ors,
            }
        )
    else:
        design_full, design_reduced = _build_design_matrices(
            meta,
            group_col=group_col,
            group_b=group_b,
            covariates=covariates,
        )

        if effective_test == "glm":
            pvalues = _glm_lrt_from_arrays(
                meth_counts,
                cov_counts,
                design_full,
                design_reduced,
                overdispersion=overdispersion,
            )
        elif effective_test == "limma":
            pvalues = _limma_ebayes_from_arrays(
                meth_counts,
                cov_counts,
                design_full,
                chunk_size=limma_chunk_size,
                alpha=limma_alpha,
            )
        else:
            raise ValueError(f"Unknown test: {test!r}")

        result = pl.DataFrame(
            {
                "chr": wide.get_column("chrom"),
                "start": (wide.get_column("pos") - 1).cast(pl.Int64),
                "end": wide.get_column("pos").cast(pl.Int64),
                "strand": wide.get_column("strand"),
                "pvalue": pvalues,
                "mean_diff": mean_diff,
                f"mean_meth_{_sanitize_group_label(group_a)}": mean_a,
                f"mean_meth_{_sanitize_group_label(group_b)}": mean_b,
            }
        )

    return result


def calculate_diff_meth_chromwise(
    methylstore: ParquetMethylStore,
    samples: Sequence[str],
    group_col: str,
    metadata: pd.DataFrame,
    *,
    test: str = "auto",
    covariates: Sequence[str] | None = None,
    overdispersion: bool = True,
    limma_chunk_size: int = 100000,
    limma_alpha: float = 0.5,
    n_threads: int | None = None,
    output_path: str | None = None,
    fdr_method: str = "BH",
) -> pd.DataFrame:
    """Run chrom-wise differential methylation and apply genome-wide FDR."""
    ordered_samples = list(samples)
    chroms = methylstore.chromosomes()
    if n_threads is None:
        n_threads = max(1, min(len(chroms), (len(chroms) or 1)))

    worker = partial(
        process_chromosome,
        methylstore,
        samples=ordered_samples,
        group_col=group_col,
        metadata=metadata,
        test=test,
        covariates=covariates,
        overdispersion=overdispersion,
        limma_chunk_size=limma_chunk_size,
        limma_alpha=limma_alpha,
    )

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        chrom_results = list(pool.map(worker, chroms))

    chrom_results = [frame for frame in chrom_results if frame is not None and frame.height > 0]
    if not chrom_results:
        return pd.DataFrame(columns=["chr", "start", "end", "strand", "pvalue", "qvalue", "mean_diff"])

    combined = pl.concat(chrom_results, how="vertical")
    qvalues = _apply_fdr(combined.get_column("pvalue").to_numpy(), method=fdr_method)
    combined = combined.with_columns(pl.Series("qvalue", qvalues))
    combined = combined.sort("pvalue")

    result = combined.to_pandas()

    if output_path is not None:
        output_path = str(output_path)
        if output_path.endswith(".parquet"):
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, sep="\t", index=False)

    return result
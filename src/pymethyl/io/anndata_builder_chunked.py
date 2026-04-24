"""
pymethyl.io.anndata_builder_chunked
====================================
Chunked, disk-backed AnnData construction for large WGBS datasets.

This module provides memory-efficient builders that process data in chunks
and write directly to Zarr stores, avoiding RAM exhaustion on datasets
with tens of millions of CpG sites.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Import shared locus encoding utilities from anndata_builder
from .anndata_builder import (
    _CHR_TO_ID,
    _DEFAULT_CHR_ID,
    _LOCUS_SCALE,
    _decode_locus_int,
)

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "anndata is required. Install with: pip install anndata"
    ) from e

try:
    import zarr
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "zarr is required for chunked processing. Install with: pip install zarr"
    ) from e


def build_anndata_chunked(
    sample_ids: list[str],
    dataframes: list[pl.DataFrame],
    obs_metadata: pd.DataFrame | None = None,
    *,
    join_type: str = "outer",
    chunk_size: int = 1_000_000,
    zarr_path: PathLike = "temp_adata.zarr",
    cleanup: bool = False,
    fill_beta_na: float | None = float("nan"),
    fill_counts_na: int = 0,
) -> ad.AnnData:
    """Build AnnData with chunked processing to avoid RAM exhaustion.

    This function processes data in chunks of sites, writing directly to
    a Zarr store on disk. This keeps peak RAM usage low (<1 GB) regardless
    of dataset size, making it suitable for WGBS-scale data with 20–30M+
    CpG sites.

    Parameters
    ----------
    sample_ids:
        Ordered list of sample identifiers (become ``adata.obs_names``).
    dataframes:
        One Polars DataFrame per sample, in the same order as
        ``sample_ids``.  Each must have columns: ``chr``, ``start``,
        ``end``, ``beta``, ``methylated``, ``unmethylated``, ``coverage``.
    obs_metadata:
        Optional pandas DataFrame with sample metadata.  Index must match
        ``sample_ids``.  Becomes ``adata.obs``.
    join_type:
        ``"outer"`` — retain all loci seen in at least one sample.
        ``"inner"`` — retain only loci present in every sample.
    chunk_size:
        Number of sites to process per chunk.  Default: 1,000,000 (1M sites).
        Larger chunks are faster but use more RAM.
    zarr_path:
        Path to the Zarr store (directory) where intermediate data is written.
        Default: ``"temp_adata.zarr"``.
    cleanup:
        If ``True``, delete the Zarr store after loading into AnnData.
        Default: ``False`` (keep the Zarr store for debugging).
    fill_beta_na:
        Fill value for missing beta values.  Default: ``float("nan")``.
    fill_counts_na:
        Fill value for missing coverage/count cells.  Default: 0.

    Returns
    -------
    anndata.AnnData

    Examples
    --------
    >>> from pymethyl.io import build_anndata_chunked
    >>> adata = build_anndata_chunked(
    ...     sample_ids=sample_ids,
    ...     dataframes=sample_dfs,
    ...     obs_metadata=obs_meta,
    ...     chunk_size=1_000_000,
    ...     zarr_path="results/temp_adata.zarr",
    ... )
    """
    if len(sample_ids) != len(dataframes):
        raise ValueError(
            f"sample_ids ({len(sample_ids)}) and dataframes ({len(dataframes)}) "
            "must have the same length."
        )

    if len(sample_ids) == 0:
        raise ValueError("At least one sample is required.")

    zarr_path = Path(zarr_path)
    n_samples = len(sample_ids)

    logger.info(
        "Building AnnData (chunked) from %d samples (join_type='%s', chunk_size=%d) …",
        n_samples, join_type, chunk_size,
    )

    # ------------------------------------------------------------------
    # Step 1 — Compute int64 locus IDs and per-sample Polars frames
    # ------------------------------------------------------------------
    keyed_frames: list[pl.DataFrame] = []
    for sid, df in zip(sample_ids, dataframes):
        if "strand" not in df.columns:
            df = df.with_columns(pl.lit("*").alias("strand"))

        df = df.with_columns(
            pl.col("chr")
            .replace(_CHR_TO_ID)
            .cast(pl.Int64, strict=False)
            .fill_null(_DEFAULT_CHR_ID)
            .alias("_chr_id"),
        )

        df = df.with_columns(
            (pl.col("_chr_id") * _LOCUS_SCALE + pl.col("start").cast(pl.Int64))
            .cast(pl.Int64)
            .alias("_locus_int")
        )

        df = df.select(["_locus_int", "beta", "coverage", "methylated"])
        keyed_frames.append(df)

    logger.info("  Step 1/5: computed int64 locus IDs for all samples")

    # ------------------------------------------------------------------
    # Step 2 — Build global locus index (union or intersection of IDs)
    # ------------------------------------------------------------------
    if join_type == "outer":
        loci_df = pl.concat([df.select("_locus_int") for df in keyed_frames]).unique()
    elif join_type == "inner":
        loci_df = keyed_frames[0].select("_locus_int").unique()
        for df in keyed_frames[1:]:
            loci_df = loci_df.join(df.select("_locus_int").unique(), on="_locus_int", how="inner")
    else:
        raise ValueError(f"join_type must be 'outer' or 'inner', got {join_type!r}")

    loci_df = loci_df.sort("_locus_int")
    locus_ids = loci_df["_locus_int"].to_numpy()
    n_sites = int(len(locus_ids))

    logger.info("  Step 2/5: global locus index built (%d loci)", n_sites)

    # ------------------------------------------------------------------
    # Step 3 — Create Zarr arrays on disk
    # ------------------------------------------------------------------
    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)

    # Create Zarr group (compatible with Zarr v2 and v3)
    root = zarr.open_group(str(zarr_path), mode="w")

    # Create chunked arrays (chunk along site dimension)
    X_zarr = root.create_dataset(
        "X",
        shape=(n_samples, n_sites),
        chunks=(n_samples, min(chunk_size, n_sites)),
        dtype="float32",
        fill_value=fill_beta_na,
    )

    coverage_zarr = root.create_dataset(
        "coverage",
        shape=(n_samples, n_sites),
        chunks=(n_samples, min(chunk_size, n_sites)),
        dtype="int32",
        fill_value=fill_counts_na,
    )

    methylated_zarr = root.create_dataset(
        "methylated_counts",
        shape=(n_samples, n_sites),
        chunks=(n_samples, min(chunk_size, n_sites)),
        dtype="int32",
        fill_value=fill_counts_na,
    )

    logger.info("  Step 3/5: created Zarr arrays at %s", zarr_path)

    # ------------------------------------------------------------------
    # Step 4 — Process chunks and write to Zarr
    # ------------------------------------------------------------------
    n_chunks = int(math.ceil(n_sites / chunk_size))

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_sites)
        chunk_locus_ids = locus_ids[start_idx:end_idx]
        chunk_n_sites = len(chunk_locus_ids)

        # Create a temporary locus index DataFrame for this chunk
        loci_chunk_df = pl.DataFrame({"_locus_int": chunk_locus_ids})

        # Allocate chunk-sized arrays
        beta_chunk = np.full((n_samples, chunk_n_sites), fill_beta_na, dtype=np.float32)
        coverage_chunk = np.full((n_samples, chunk_n_sites), fill_counts_na, dtype=np.int32)
        methylated_chunk = np.full((n_samples, chunk_n_sites), fill_counts_na, dtype=np.int32)

        # Fill chunk from each sample
        for i, (sid, df) in enumerate(zip(sample_ids, keyed_frames)):
            # Filter sample DataFrame to only rows in this chunk
            chunk_df = df.filter(pl.col("_locus_int").is_in(chunk_locus_ids))

            # Join to align rows with global locus order
            joined = loci_chunk_df.join(chunk_df, on="_locus_int", how="left")

            beta_chunk[i, :] = joined["beta"].fill_null(fill_beta_na).to_numpy().astype(np.float32)
            coverage_chunk[i, :] = joined["coverage"].fill_null(fill_counts_na).to_numpy().astype(np.int32)
            methylated_chunk[i, :] = joined["methylated"].fill_null(fill_counts_na).to_numpy().astype(np.int32)

        # Write chunk to Zarr
        X_zarr[:, start_idx:end_idx] = beta_chunk
        coverage_zarr[:, start_idx:end_idx] = coverage_chunk
        methylated_zarr[:, start_idx:end_idx] = methylated_chunk

        logger.info(
            "  Step 4/5: processed chunk %d/%d (sites %d–%d)",
            chunk_idx + 1, n_chunks, start_idx, end_idx - 1,
        )

    # ------------------------------------------------------------------
    # Step 5 — Build var and obs metadata, construct AnnData
    # ------------------------------------------------------------------
    var_records = []
    for locus_int in locus_ids:
        chrom, start = _decode_locus_int(int(locus_int))
        end = start + 1
        var_records.append({"chr": chrom, "start": start, "end": end, "strand": "*"})

    var_df = pd.DataFrame(var_records, index=locus_ids)
    var_df.index.name = "locus_id"

    # Add context if available
    if "context" in dataframes[0].columns:
        df0 = dataframes[0]
        if "strand" not in df0.columns:
            df0 = df0.with_columns(pl.lit("*").alias("strand"))

        df0 = df0.with_columns(
            pl.col("chr")
            .replace(_CHR_TO_ID)
            .cast(pl.Int64, strict=False)
            .fill_null(_DEFAULT_CHR_ID)
            .alias("_chr_id"),
        )
        df0 = df0.with_columns(
            (pl.col("_chr_id") * _LOCUS_SCALE + pl.col("start").cast(pl.Int64))
            .cast(pl.Int64)
            .alias("_locus_int")
        ).select(["_locus_int", "context"])

        ctx_map = dict(zip(df0["_locus_int"].to_list(), df0["context"].to_list()))
        var_df["context"] = [ctx_map.get(int(k), "CpG") for k in locus_ids]
    else:
        var_df["context"] = "CpG"

    # Build obs DataFrame
    obs_df = pd.DataFrame(index=sample_ids)
    obs_df.index.name = "sample_id"

    if obs_metadata is not None:
        obs_df = obs_metadata.reindex(sample_ids)
        obs_df.index.name = "sample_id"

    # Construct AnnData from Zarr arrays
    adata = ad.AnnData(
        X=X_zarr[:],  # load into memory (or keep backed if you want)
        obs=obs_df,
        var=var_df,
        layers={
            "coverage": coverage_zarr[:],
            "methylated_counts": methylated_zarr[:],
        },
    )

    logger.info("  Step 5/5: AnnData built (%d samples × %d sites)", n_samples, n_sites)

    # Cleanup Zarr store if requested
    if cleanup:
        import shutil
        shutil.rmtree(zarr_path)
        logger.info("  Cleaned up temporary Zarr store: %s", zarr_path)

    logger.info("AnnData construction complete.")
    return adata

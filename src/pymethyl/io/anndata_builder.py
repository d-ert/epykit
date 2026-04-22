"""
pymethyl.io.anndata_builder
===========================
Constructs AnnData objects from per-sample Polars DataFrames and provides
Zarr/HDF5 persistence helpers.

AnnData geometry
----------------
AnnData strictly enforces: **observations (samples) = rows**, **variables
(CpG sites) = columns**.  This is the *opposite* of the methylKit/
SummarizedExperiment convention — be aware when porting R code.

  - ``adata.X``                          : beta-value matrix  (n_samples × n_sites)
  - ``adata.obs``                        : sample metadata DataFrame
  - ``adata.var``                        : site coordinate DataFrame
  - ``adata.layers['coverage']``         : total read coverage matrix
  - ``adata.layers['methylated_counts']``: methylated read count matrix

Locus key
---------
Sites are uniquely identified by the composite key: ``chr:start-end:strand``.
This string is used as the ``var_names`` index.

Memory notes
------------
The beta matrix is stored as a NumPy float32 array (or sparse array for
very sparse tissues).  Coverage and methylated counts are stored as int32.
NaN values represent missing coverage at a locus in a given sample.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "anndata is required for AnnData construction. "
        "Install it with: pip install anndata"
    ) from e


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_anndata(
    sample_ids: list[str],
    dataframes: list[pl.DataFrame],
    obs_metadata: pd.DataFrame | None = None,
    *,
    join_type: str = "outer",
    fill_beta_na: float | None = float("nan"),
    fill_counts_na: int = 0,
    sparse: bool = False,
) -> ad.AnnData:
    """Build a cohort-level AnnData object from per-sample Polars DataFrames.

    Parameters
    ----------
    sample_ids:
        Ordered list of sample identifiers (become ``adata.obs_names``).
    dataframes:
        One Polars DataFrame per sample, in the same order as
        ``sample_ids``.  Each must have columns: ``chr``, ``start``,
        ``end``, ``strand``, ``beta``, ``methylated``, ``unmethylated``,
        ``coverage``.
    obs_metadata:
        Optional pandas DataFrame with sample metadata.  Index must match
        ``sample_ids``.  Becomes ``adata.obs``.
    join_type:
        ``"outer"`` — retain all loci seen in at least one sample (missing
        values filled with NaN).  ``"inner"`` — retain only loci present in
        every sample.
    fill_beta_na:
        Fill value for missing beta values.  Default: ``float("nan")``.
    fill_counts_na:
        Fill value for missing coverage/count cells.  Default: 0.
    sparse:
        If ``True``, store the beta matrix as a sparse array (useful when
        large portions of the genome have zero methylation).

    Returns
    -------
    anndata.AnnData

    Notes
    -----
    The function performs these steps:

    1. Construct a string locus key ``chr:start-end:strand`` for each site.
    2. For each sample, create a mapping ``{locus_key -> (beta, coverage, methylated)}``.
    3. Build the union (or intersection) of all locus keys.
    4. Assemble dense matrices of shape ``(n_samples, n_sites)``.
    5. Wrap into AnnData with obs / var / layers.
    """
    if len(sample_ids) != len(dataframes):
        raise ValueError(
            f"sample_ids ({len(sample_ids)}) and dataframes ({len(dataframes)}) "
            "must have the same length."
        )

    if len(sample_ids) == 0:
        raise ValueError("At least one sample is required.")

    logger.info(
        "Building AnnData from %d samples (join_type='%s') …",
        len(sample_ids), join_type,
    )

    # ------------------------------------------------------------------
    # Step 1 — Compute locus keys and per-sample Polars frames
    # ------------------------------------------------------------------
    keyed_frames: list[pl.DataFrame] = []
    for sid, df in zip(sample_ids, dataframes):
        # Build locus key
        df = df.with_columns(
            (
                pl.col("chr").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col("start").cast(pl.Utf8)
                + pl.lit("-")
                + pl.col("end").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col("strand").cast(pl.Utf8)
            ).alias("_locus_key")
        )
        # Keep only the columns we need
        keep_cols = ["_locus_key", "beta", "coverage", "methylated"]
        # Rename to sample-specific names so we can join
        df = df.select(keep_cols).rename({
            "beta": f"beta__{sid}",
            "coverage": f"coverage__{sid}",
            "methylated": f"methylated__{sid}",
        })
        keyed_frames.append(df)

    # ------------------------------------------------------------------
    # Step 2 — Join all samples on locus key
    # ------------------------------------------------------------------
    combined = keyed_frames[0]
    for df in keyed_frames[1:]:
        combined = combined.join(df, on="_locus_key", how=join_type)

    logger.info("  Combined matrix: %d loci", len(combined))

    # Sort loci for deterministic output (chr, then start numerically)
    # Parse locus key back to sort columns
    combined = combined.with_columns([
        pl.col("_locus_key").str.split(":").list.get(0).alias("_sort_chr"),
        pl.col("_locus_key")
            .str.split(":")
            .list.get(1)
            .str.split("-")
            .list.get(0)
            .cast(pl.Int64)
            .alias("_sort_start"),
    ])
    combined = combined.sort(["_sort_chr", "_sort_start"]).drop(["_sort_chr", "_sort_start"])

    locus_keys = combined["_locus_key"].to_list()
    n_sites = len(locus_keys)

    # ------------------------------------------------------------------
    # Step 3 — Assemble matrices (n_samples × n_sites)
    # ------------------------------------------------------------------
    beta_mat = np.full((len(sample_ids), n_sites), fill_beta_na, dtype=np.float32)
    coverage_mat = np.full((len(sample_ids), n_sites), fill_counts_na, dtype=np.int32)
    methylated_mat = np.full((len(sample_ids), n_sites), fill_counts_na, dtype=np.int32)

    for i, sid in enumerate(sample_ids):
        beta_col = f"beta__{sid}"
        cov_col = f"coverage__{sid}"
        meth_col = f"methylated__{sid}"

        if beta_col in combined.columns:
            beta_mat[i, :] = combined[beta_col].fill_null(fill_beta_na).to_numpy().astype(np.float32)
        if cov_col in combined.columns:
            coverage_mat[i, :] = combined[cov_col].fill_null(fill_counts_na).to_numpy().astype(np.int32)
        if meth_col in combined.columns:
            methylated_mat[i, :] = combined[meth_col].fill_null(fill_counts_na).to_numpy().astype(np.int32)

    if sparse:
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.nan_to_num(beta_mat, nan=0.0))
    else:
        X = beta_mat

    # ------------------------------------------------------------------
    # Step 4 — Build var DataFrame from locus keys
    # ------------------------------------------------------------------
    # Parse locus keys: "chr1:1000-1001:+"
    var_records = []
    for key in locus_keys:
        parts = key.split(":")
        chrom = parts[0]
        strand = parts[2] if len(parts) > 2 else "*"
        pos_parts = parts[1].split("-")
        start = int(pos_parts[0])
        end = int(pos_parts[1]) if len(pos_parts) > 1 else start + 1
        var_records.append({"chr": chrom, "start": start, "end": end, "strand": strand})

    var_df = pd.DataFrame(var_records, index=locus_keys)
    var_df.index.name = "locus_key"

    # Add context from first sample that has coverage at that site
    # (simplified: use the context from sample 0's frame if available)
    if "context" in dataframes[0].columns:
        df0 = dataframes[0].with_columns(
            (
                pl.col("chr").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col("start").cast(pl.Utf8)
                + pl.lit("-")
                + pl.col("end").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col("strand").cast(pl.Utf8)
            ).alias("_locus_key")
        ).select(["_locus_key", "context"])
        ctx_map = dict(zip(df0["_locus_key"].to_list(), df0["context"].to_list()))
        var_df["context"] = [ctx_map.get(k, "CpG") for k in locus_keys]
    else:
        var_df["context"] = "CpG"

    # ------------------------------------------------------------------
    # Step 5 — Build obs DataFrame
    # ------------------------------------------------------------------
    obs_df = pd.DataFrame(index=sample_ids)
    obs_df.index.name = "sample_id"

    if obs_metadata is not None:
        # Align to sample_ids order
        obs_df = obs_metadata.reindex(sample_ids)
        obs_df.index.name = "sample_id"

    # ------------------------------------------------------------------
    # Step 6 — Construct AnnData
    # ------------------------------------------------------------------
    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=var_df,
        layers={
            "coverage": coverage_mat,
            "methylated_counts": methylated_mat,
        },
    )

    logger.info(
        "AnnData built: %d samples × %d sites", adata.n_obs, adata.n_vars
    )
    return adata


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save(adata: ad.AnnData, path: PathLike, format: str = "zarr") -> None:
    """Persist an AnnData object to disk.

    Parameters
    ----------
    adata:
        The AnnData object to save.
    path:
        Output path.  For Zarr, this is a directory.  For HDF5, use a
        ``.h5ad`` extension.
    format:
        ``"zarr"`` (recommended for large cohorts, chunked + compressed) or
        ``"h5ad"`` (single HDF5 file, compatible with older tools).

    Examples
    --------
    >>> save(adata, "cohort.zarr")
    >>> save(adata, "cohort.h5ad", format="h5ad")
    """
    path = Path(path)
    if format == "zarr":
        adata.write_zarr(str(path))
        logger.info("Saved AnnData to Zarr store: %s", path)
    elif format in ("h5ad", "hdf5"):
        if not str(path).endswith(".h5ad"):
            path = path.with_suffix(".h5ad")
        adata.write_h5ad(str(path))
        logger.info("Saved AnnData to HDF5: %s", path)
    else:
        raise ValueError(f"Unsupported format: '{format}'. Use 'zarr' or 'h5ad'.")


def load(
    path: PathLike,
    backed: str | None = None,
    format: str | None = None,
) -> ad.AnnData:
    """Load an AnnData object from disk.

    Parameters
    ----------
    path:
        Path to the stored AnnData (Zarr directory or ``.h5ad`` file).
    backed:
        For lazy / out-of-core access:
        ``"r"`` (read-only backed mode) or ``"r+"`` (backed read-write).
        Only supported for HDF5 (``.h5ad``) files.  Zarr is always lazily
        streamed.  Default: ``None`` (fully load into RAM).
    format:
        Force format: ``"zarr"`` or ``"h5ad"``.  If ``None``, auto-detected
        from path.

    Returns
    -------
    anndata.AnnData

    Examples
    --------
    >>> adata = load("cohort.zarr")
    >>> adata_lazy = load("cohort.h5ad", backed="r")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData store not found: {path}")

    # Auto-detect format
    if format is None:
        if path.suffix == ".h5ad":
            format = "h5ad"
        elif path.is_dir():
            format = "zarr"
        else:
            format = "h5ad"

    if format == "zarr":
        adata = ad.read_zarr(str(path))
        logger.info("Loaded AnnData from Zarr: %s  (%d × %d)", path, adata.n_obs, adata.n_vars)
    elif format in ("h5ad", "hdf5"):
        adata = ad.read_h5ad(str(path), backed=backed)
        logger.info("Loaded AnnData from HDF5: %s  (%d × %d)", path, adata.n_obs, adata.n_vars)
    else:
        raise ValueError(f"Unsupported format: '{format}'.")

    return adata

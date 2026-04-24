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

# ---------------------------------------------------------------------------
# Locus key encoding
# ---------------------------------------------------------------------------

# We encode genomic loci as compact int64 identifiers to avoid the RAM overhead
# of Python strings like "chr1:123456-123457:*".  For human-scale genomes with
# positions < 1e10, this scheme is safe and provides a natural sort order:
#   locus_int = chr_id * SCALE + start
# where chr_id is a small integer code for each chromosome.

_CHR_TO_ID: dict[str, int] = {f"chr{i}": i for i in range(1, 23)}
_CHR_TO_ID.update({"chrX": 23, "chrY": 24, "chrM": 25, "chrMT": 25})
_ID_TO_CHR: dict[int, str] = {v: k for k, v in _CHR_TO_ID.items()}

_LOCUS_SCALE: int = 10_000_000_000  # must be > max genomic position
_DEFAULT_CHR_ID: int = 99  # fallback for unexpected chromosome names


def _decode_locus_int(locus_int: int) -> tuple[str, int]:
    """Decode an encoded int64 locus back to (chromosome, start).

    Unknown chromosome IDs are mapped to "chrUn".
    """

    chr_id = locus_int // _LOCUS_SCALE
    start = int(locus_int % _LOCUS_SCALE)
    chrom = _ID_TO_CHR.get(int(chr_id), "chrUn")
    return chrom, start

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
        ``end``, ``beta``, ``methylated``, ``unmethylated``,
        ``coverage``.  A ``strand`` column is optional — if missing,
        it is treated as ``"*"`` (unknown strand) when constructing
        locus keys.
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
    # Step 1 — Compute int64 locus IDs and per-sample Polars frames
    # ------------------------------------------------------------------
    # We avoid string locus keys ("chr1:1000-1001:*") and instead encode
    # loci as compact int64 identifiers: chr_id * SCALE + start.  This
    # dramatically reduces memory usage for 10–30M loci and makes joins
    # and sorting much faster.
    keyed_frames: list[pl.DataFrame] = []
    for sid, df in zip(sample_ids, dataframes):
        # Ensure a strand column exists; Bismark .cov files, for example,
        # do not carry strand information. Treat missing strand as "*".
        if "strand" not in df.columns:
            df = df.with_columns(pl.lit("*").alias("strand"))

        # Map chromosome strings to small integer IDs, falling back to a
        # generic "chrUn" bucket for unexpected contig names.
        df = df.with_columns(
            # Map chromosome strings to small integer IDs using a dictionary.
            # We use ``replace`` (widely supported in Polars) followed by a
            # non-strict cast so that unknown contigs become null, then fill
            # them with a fallback ID.
            pl.col("chr")
            .replace(_CHR_TO_ID)
            .cast(pl.Int64, strict=False)
            .fill_null(_DEFAULT_CHR_ID)
            .alias("_chr_id"),
        )

        # Build int64 locus ID: chr_id * SCALE + start
        df = df.with_columns(
            (pl.col("_chr_id") * _LOCUS_SCALE + pl.col("start").cast(pl.Int64))
            .cast(pl.Int64)
            .alias("_locus_int")
        )

        # Keep only the columns we need for matrix construction
        df = df.select(["_locus_int", "beta", "coverage", "methylated"])
        keyed_frames.append(df)

    logger.info("  Step 1/3: computed int64 locus IDs for all samples")

    # ------------------------------------------------------------------
    # Step 2 — Build global locus index (union or intersection of IDs)
    # ------------------------------------------------------------------
    if join_type == "outer":
        # Union of all locus IDs across samples
        loci_df = pl.concat([df.select("_locus_int") for df in keyed_frames]).unique()
    elif join_type == "inner":
        # Intersection of locus IDs across samples, implemented as a series
        # of key-only inner joins. This is much cheaper than joining full
        # matrices.
        loci_df = keyed_frames[0].select("_locus_int").unique()
        for df in keyed_frames[1:]:
            loci_df = loci_df.join(df.select("_locus_int").unique(), on="_locus_int", how="inner")
    else:
        raise ValueError(f"join_type must be 'outer' or 'inner', got {join_type!r}")

    # Sort loci for deterministic output. Because our encoding is
    # chr_id * SCALE + start, a simple numeric sort yields (chr, start)
    # ordering without needing to materialise chromosome strings.
    loci_df = loci_df.sort("_locus_int")

    # Materialise ordered locus IDs as a NumPy array for downstream use
    locus_ids = loci_df["_locus_int"].to_numpy()
    n_sites = int(len(locus_ids))

    logger.info("  Step 2/3: global locus index built (%d loci)", n_sites)

    # ------------------------------------------------------------------
    # Step 3 — Assemble matrices (n_samples × n_sites)
    # ------------------------------------------------------------------
    n_samples = len(sample_ids)

    # Coverage and methylated counts are stored densely as before.
    coverage_mat = np.full((n_samples, n_sites), fill_counts_na, dtype=np.int32)
    methylated_mat = np.full((n_samples, n_sites), fill_counts_na, dtype=np.int32)

    if sparse:
        # Build a sparse beta matrix, which is typically much smaller for
        # union joins where many sites are missing in some samples.
        from scipy.sparse import lil_matrix

        beta_sparse = lil_matrix((n_samples, n_sites), dtype=np.float32)
    else:
        beta_mat = np.full((n_samples, n_sites), fill_beta_na, dtype=np.float32)

    # For each sample, left-join onto the global locus index to align rows,
    # then fill the corresponding row in the matrices.
    for i, (sid, df) in enumerate(zip(sample_ids, keyed_frames)):
        joined = loci_df.join(df, on="_locus_int", how="left")

        # Fill coverage / methylated counts (dense)
        coverage_mat[i, :] = (
            joined["coverage"].fill_null(fill_counts_na).to_numpy().astype(np.int32)
        )
        methylated_mat[i, :] = (
            joined["methylated"].fill_null(fill_counts_na).to_numpy().astype(np.int32)
        )

        # Fill beta values (dense or sparse)
        beta_col = joined["beta"]
        if sparse:
            # Encode only non-null beta values in the sparse matrix
            beta_vals = beta_col.fill_null(np.nan).to_numpy().astype(np.float32)
            mask = ~np.isnan(beta_vals)
            if mask.any():
                cols = np.nonzero(mask)[0]
                beta_sparse[i, cols] = beta_vals[mask]
        else:
            beta_mat[i, :] = beta_col.fill_null(fill_beta_na).to_numpy().astype(np.float32)

        logger.info("    Filled matrices for sample %s (%d/%d)", sid, i + 1, n_samples)

    if sparse:
        from scipy.sparse import csr_matrix

        X = csr_matrix(beta_sparse)
    else:
        X = beta_mat

    # ------------------------------------------------------------------
    # Step 4 — Build var DataFrame from encoded locus IDs
    # ------------------------------------------------------------------
    # Decode int64 locus IDs back to (chr, start).  We default to strand "*"
    # and infer end as start + 1, which matches typical CpG coverage formats.
    var_records = []
    for locus_int in locus_ids:
        chrom, start = _decode_locus_int(int(locus_int))
        end = start + 1
        var_records.append({"chr": chrom, "start": start, "end": end, "strand": "*"})

    var_df = pd.DataFrame(var_records, index=locus_ids)
    var_df.index.name = "locus_id"

    # Add context from first sample that has coverage at that site
    # (simplified: use the context from sample 0's frame if available)
    if "context" in dataframes[0].columns:
        df0 = dataframes[0]
        if "strand" not in df0.columns:
            df0 = df0.with_columns(pl.lit("*").alias("strand"))

        # Recompute int64 locus IDs for the first sample to map context
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

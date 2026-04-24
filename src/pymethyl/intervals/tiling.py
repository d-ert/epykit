"""
pymethyl.intervals.tiling
==========================
Genomic interval operations: tiling windows, feature annotation, and
CpG island classification.

Backend strategy
----------------
1. **polars-bio** (primary) — Rust/DataFusion-powered, 38–282× faster
   than bioframe on overlap/coverage operations; streaming for
   out-of-core datasets.
2. **Pure-Polars fallback** — if polars-bio is not installed, a pure
   Polars implementation is used.  It is correct but ~10× slower for
   very large datasets.

The public API is identical regardless of which backend is active.
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
    raise ImportError("anndata is required: pip install anndata") from e

# Optional polars-bio
try:
    import polars_bio as pb  # type: ignore
    _HAS_POLARS_BIO = True
    logger.debug("polars-bio backend active for interval operations.")
except ImportError:
    _HAS_POLARS_BIO = False
    logger.debug("polars-bio not found — using pure-Polars fallback.")


# ---------------------------------------------------------------------------
# Tiling windows
# ---------------------------------------------------------------------------

def tile_counts(
    adata: "ad.AnnData",
    window: int = 1000,
    step: int | None = None,
    min_cpgs_per_window: int = 1,
) -> "ad.AnnData":
    """Bin CpG sites into fixed-size genomic windows.

    Replicates ``methylKit::tileMethylCounts(tileSize, stepSize)``.
    For each window, sums the ``methylated_counts`` and ``coverage``
    layers across all CpG sites within the window, then recomputes beta
    as the window-level methylation percentage.

    Parameters
    ----------
    adata:
        Input AnnData (single-base resolution).  Must have layers
        ``coverage`` and ``methylated_counts``, and ``var`` columns
        ``chr``, ``start``, ``end``.
    window:
        Genomic window size in base pairs.  Default: 1000.
    step:
        Step size (stride) between windows in bp.  If ``None``, defaults
        to ``window`` (non-overlapping tiles).  Use ``step < window``
        for sliding windows.
    min_cpgs_per_window:
        Minimum number of CpG sites required to include a window in the
        output.  Default: 1 (keep all windows with at least 1 site).

    Returns
    -------
    anndata.AnnData
        Tiled AnnData object:
          - Same samples (obs)
          - New variables (var): window loci (chr, start, end)
          - ``X``: beta per window
          - ``layers['coverage']``: summed coverage per window
          - ``layers['methylated_counts']``: summed methylated counts per window
          - ``var['n_cpgs']``: number of CpG sites in each window

    Examples
    --------
    >>> tiled = tile_counts(adata, window=1000)
    >>> tiled_sliding = tile_counts(adata, window=1000, step=500)
    """
    if step is None:
        step = window

    # Extract site coordinates from var
    sites_pl = pl.from_pandas(
        adata.var[["chr", "start", "end"]].reset_index()
    ).rename({"locus_key": "_site_key"})

    # --- Generate tiles ---
    # Get chromosome sizes from the data (max end position per chr)
    chr_sizes = (
        sites_pl
        .group_by("chr")
        .agg(pl.col("end").max().alias("chr_size"))
    )

    tiles = _generate_tiles(chr_sizes, window, step)
    logger.info(
        "tile_counts(window=%d, step=%d): %d tiles generated",
        window, step, len(tiles),
    )

    # --- Overlap sites with tiles ---
    if _HAS_POLARS_BIO:
        site_to_tile = _overlap_polars_bio(sites_pl, tiles)
    else:
        site_to_tile = _overlap_pure_polars(sites_pl, tiles)

    if len(site_to_tile) == 0:
        logger.warning("No CpG sites overlap with any tile. Returning empty AnnData.")
        return _empty_anndata(adata)

    # --- Filter tiles by min_cpgs ---
    tile_cpg_counts = site_to_tile.group_by("_tile_key").agg(
        pl.count().alias("n_cpgs")
    )
    valid_tiles = tile_cpg_counts.filter(pl.col("n_cpgs") >= min_cpgs_per_window)
    site_to_tile = site_to_tile.join(valid_tiles.select("_tile_key"), on="_tile_key", how="inner")

    # Map site index: locus_key -> position index in adata.var
    site_idx_map = {k: i for i, k in enumerate(adata.var_names)}
    site_to_tile = site_to_tile.with_columns(
        pl.col("_site_key").map_elements(
            lambda k: site_idx_map.get(k, -1), return_dtype=pl.Int32
        ).alias("_site_idx")
    ).filter(pl.col("_site_idx") >= 0)

    # Get unique tiles sorted by genomic position
    unique_tiles = (
        tiles
        .join(valid_tiles.select("_tile_key"), on="_tile_key", how="inner")
        .sort(["chr", "start"])
    )

    tile_idx_map = {k: i for i, k in enumerate(unique_tiles["_tile_key"].to_list())}
    n_tiles = len(unique_tiles)
    n_samples = adata.n_obs

    # --- Aggregate matrices ---
    cov_arr = np.asarray(adata.layers["coverage"])      # (n_samples, n_sites)
    meth_arr = np.asarray(adata.layers["methylated_counts"])

    tile_cov = np.zeros((n_samples, n_tiles), dtype=np.int32)
    tile_meth = np.zeros((n_samples, n_tiles), dtype=np.int32)

    s2t = site_to_tile.select(["_site_idx", "_tile_key"]).to_pandas()
    for _, row in s2t.iterrows():
        s_i = int(row["_site_idx"])
        t_i = tile_idx_map.get(row["_tile_key"], -1)
        if t_i < 0:
            continue
        tile_cov[:, t_i] += cov_arr[:, s_i]
        tile_meth[:, t_i] += meth_arr[:, s_i]

    # Compute beta for tiles
    with np.errstate(divide="ignore", invalid="ignore"):
        tile_beta = np.where(
            tile_cov > 0,
            tile_meth.astype(np.float32) / tile_cov.astype(np.float32) * 100.0,
            np.nan,
        ).astype(np.float32)

    # --- Build var DataFrame for tiles ---
    tile_pd = unique_tiles.to_pandas()
    n_cpg_map = dict(zip(
        tile_cpg_counts["_tile_key"].to_list(),
        tile_cpg_counts["n_cpgs"].to_list(),
    ))
    tile_pd["n_cpgs"] = tile_pd["_tile_key"].map(n_cpg_map).fillna(0).astype(int)
    tile_pd = tile_pd.set_index("_tile_key")
    tile_pd.index.name = "locus_key"

    # --- Build new AnnData ---
    tiled_adata = ad.AnnData(
        X=tile_beta,
        obs=adata.obs.copy(),
        var=tile_pd[["chr", "start", "end", "n_cpgs"]],
        layers={
            "coverage": tile_cov,
            "methylated_counts": tile_meth,
        },
    )

    logger.info(
        "tile_counts: produced %d × %d tiled AnnData", n_samples, n_tiles
    )
    return tiled_adata


# ---------------------------------------------------------------------------
# Feature annotation
# ---------------------------------------------------------------------------

def annotate_features(
    adata: "ad.AnnData",
    bed_file: PathLike,
    *,
    feature_col: str = "feature",
    name_col: str | None = None,
    inplace: bool = False,
) -> "ad.AnnData":
    """Annotate CpG sites by overlap with a BED file of genomic features.

    Parameters
    ----------
    adata:
        Input AnnData with ``var`` columns ``chr``, ``start``, ``end``.
    bed_file:
        Path to a BED file (3+ columns: chr, start, end [, name, ...]).
        Can be gzip-compressed.
    feature_col:
        Column name to add to ``adata.var`` storing the overlapping feature
        name.  Default: ``"feature"``.
    name_col:
        If the BED file has a 4th column with feature names (e.g. gene names),
        specify the column index or name here.  Default: ``None`` (uses the
        BED region coordinates as label).
    inplace:
        If ``True``, modify ``adata.var`` in place.  If ``False`` (default),
        return a modified copy.

    Returns
    -------
    anndata.AnnData
        With ``adata.var[feature_col]`` added (``None`` for non-overlapping
        sites).

    Examples
    --------
    >>> adata_ann = annotate_features(adata, "hg38_promoters.bed",
    ...                               feature_col="promoter")
    """
    bed_file = Path(bed_file)
    if not bed_file.exists():
        raise FileNotFoundError(f"BED file not found: {bed_file}")

    # Read BED file — read without renaming first, then rename only what exists
    bed = pl.read_csv(
        str(bed_file),
        separator="\t",
        has_header=False,
        comment_prefix="#",
        infer_schema_length=50,
    )
    # Rename first columns to canonical names (bed may have 3, 4, or more cols)
    rename_map = {
        bed.columns[0]: "chr",
        bed.columns[1]: "start",
        bed.columns[2]: "end",
    }
    bed = bed.rename(rename_map)
    # Keep only needed columns
    if bed.width > 3 and name_col is not None:
        n_col = f"col_{name_col}" if isinstance(name_col, int) else name_col
        bed = bed.select(["chr", "start", "end", n_col]).rename({n_col: "feature_name"})
    else:
        bed = bed.select(["chr", "start", "end"])
        bed = bed.with_columns(
            (pl.col("chr") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("feature_name")
        )

    # Sites as Polars DataFrame
    sites = pl.from_pandas(adata.var[["chr", "start", "end"]].reset_index()).rename(
        {"locus_key": "_site_key"}
    )

    if _HAS_POLARS_BIO:
        overlap = pb.overlap(
            sites.rename({"chr": "chrom1", "start": "start1", "end": "end1"}),
            bed.rename({"chr": "chrom2", "start": "start2", "end": "end2"}),
            suffixes=("", "_feat"),
        )
        overlap = overlap.select(["_site_key", "feature_name"])
    else:
        overlap = _overlap_annotation_pure_polars(sites, bed)

    # Deduplicate (take first hit per site)
    overlap = overlap.unique(subset=["_site_key"], keep="first")
    feat_map = dict(zip(
        overlap["_site_key"].to_list(),
        overlap["feature_name"].to_list(),
    ))

    result = adata if inplace else adata.copy()
    result.var[feature_col] = [feat_map.get(k, None) for k in result.var_names]

    n_annotated = sum(1 for v in result.var[feature_col] if v is not None)
    logger.info(
        "annotate_features: %d / %d sites annotated with '%s'",
        n_annotated, adata.n_vars, feature_col,
    )
    return result


# ---------------------------------------------------------------------------
# CpG island annotation
# ---------------------------------------------------------------------------

def annotate_cpg_islands(
    adata: "ad.AnnData",
    cpg_island_bed: PathLike,
    *,
    shore_distance: int = 2000,
    shelf_distance: int = 4000,
    region_col: str = "cpg_region",
    inplace: bool = False,
) -> "ad.AnnData":
    """Classify CpG sites as island / shore / shelf / open sea.

    CpG islands are defined by the provided BED file (e.g. UCSC CpG island
    track).  The classification follows standard epigenomics conventions:
      - **Island**   : overlaps a CpG island
      - **Shore**    : within ``shore_distance`` bp of an island edge
      - **Shelf**    : between ``shore_distance`` and ``shelf_distance`` bp
      - **Open Sea** : everything else

    Parameters
    ----------
    adata:
        Input AnnData with site coordinates in ``var``.
    cpg_island_bed:
        Path to BED file of CpG island coordinates.
    shore_distance:
        Distance in bp defining the shore zone.  Default: 2000 (UCSC standard).
    shelf_distance:
        Distance in bp defining the shelf outer edge.  Default: 4000.
    region_col:
        Name of the new column added to ``adata.var``.  Default: ``"cpg_region"``.
    inplace:
        Modify in place if ``True``.

    Returns
    -------
    anndata.AnnData
        With ``adata.var[region_col]`` containing one of:
        ``"island"``, ``"shore"``, ``"shelf"``, ``"open_sea"``.

    Examples
    --------
    >>> adata_cpgi = annotate_cpg_islands(adata, "hg38_cpg_islands.bed")
    >>> adata.var["cpg_region"].value_counts()
    """
    cpg_island_bed = Path(cpg_island_bed)
    if not cpg_island_bed.exists():
        raise FileNotFoundError(f"CpG island BED file not found: {cpg_island_bed}")

    islands_raw = pl.read_csv(
        str(cpg_island_bed),
        separator="\t",
        has_header=False,
        comment_prefix="#",
        infer_schema_length=50,
    )
    islands = islands_raw.rename({
        islands_raw.columns[0]: "chr",
        islands_raw.columns[1]: "start",
        islands_raw.columns[2]: "end",
    }).select(["chr", "start", "end"])

    # Create shore and shelf intervals by expanding the island intervals
    shores = islands.with_columns([
        (pl.col("start") - shore_distance).clip(lower_bound=0).alias("start"),
        (pl.col("end") + shore_distance).alias("end"),
    ])
    shelves = islands.with_columns([
        (pl.col("start") - shelf_distance).clip(lower_bound=0).alias("start"),
        (pl.col("end") + shelf_distance).alias("end"),
    ])

    sites = pl.from_pandas(adata.var[["chr", "start", "end"]].reset_index()).rename(
        {"locus_key": "_site_key"}
    )

    def _get_overlapping_sites(sites_df: pl.DataFrame, regions: pl.DataFrame) -> set:
        """Return set of site keys that overlap regions."""
        result = _overlap_pure_polars(sites_df, regions)
        return set(result["_site_key"].to_list())

    island_keys = _get_overlapping_sites(sites, islands)
    shore_keys = _get_overlapping_sites(sites, shores) - island_keys
    shelf_keys = _get_overlapping_sites(sites, shelves) - island_keys - shore_keys

    def classify(key: str) -> str:
        if key in island_keys:
            return "island"
        elif key in shore_keys:
            return "shore"
        elif key in shelf_keys:
            return "shelf"
        return "open_sea"

    result = adata if inplace else adata.copy()
    result.var[region_col] = [classify(k) for k in result.var_names]

    counts = pd.Series(result.var[region_col]).value_counts()
    logger.info("annotate_cpg_islands: %s", counts.to_dict())
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_tiles(
    chr_sizes: pl.DataFrame,
    window: int,
    step: int,
) -> pl.DataFrame:
    """Generate genomic tiles from chromosome sizes."""
    tiles_list = []
    for row in chr_sizes.iter_rows(named=True):
        chrom = row["chr"]
        size = row["chr_size"]
        starts = range(0, size, step)
        for s in starts:
            e = min(s + window, size)
            tile_key = f"{chrom}:{s}-{e}:*"
            tiles_list.append({"chr": chrom, "start": s, "end": e, "_tile_key": tile_key})
    return pl.DataFrame(tiles_list)


def _overlap_polars_bio(
    sites: pl.DataFrame,
    tiles: pl.DataFrame,
) -> pl.DataFrame:
    """Overlap sites with tiles using polars-bio."""
    result = pb.overlap(
        sites.rename({"chr": "chrom1", "start": "start1", "end": "end1"}),
        tiles.rename({"chr": "chrom2", "start": "start2", "end": "end2"}),
        suffixes=("", "_tile"),
    )
    return result.select(["_site_key", "_tile_key"])


def _overlap_pure_polars(
    sites: pl.DataFrame,
    tiles: pl.DataFrame,
) -> pl.DataFrame:
    """Pure-Polars interval overlap using cross-join + filter."""
    # For large datasets this is slow; polars-bio is strongly preferred
    joined = sites.join(tiles, on="chr", how="inner", suffix="_tile")
    overlapping = joined.filter(
        (pl.col("start") < pl.col("end_tile"))
        & (pl.col("end") > pl.col("start_tile"))
    )

    # Build _site_key and _tile_key if not present
    if "_site_key" not in overlapping.columns:
        overlapping = overlapping.with_columns(
            (pl.col("chr") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8) + ":*").alias("_site_key")
        )
    if "_tile_key" not in overlapping.columns:
        overlapping = overlapping.with_columns(
            (pl.col("chr") + ":" + pl.col("start_tile").cast(pl.Utf8)
             + "-" + pl.col("end_tile").cast(pl.Utf8) + ":*").alias("_tile_key")
        )

    return overlapping.select(["_site_key", "_tile_key"])


def _overlap_annotation_pure_polars(
    sites: pl.DataFrame,
    bed: pl.DataFrame,
) -> pl.DataFrame:
    """Pure-Polars overlap for feature annotation."""
    joined = sites.join(bed, on="chr", how="inner", suffix="_feat")
    overlapping = joined.filter(
        (pl.col("start") < pl.col("end_feat"))
        & (pl.col("end") > pl.col("start_feat"))
    )
    if "_site_key" not in overlapping.columns:
        overlapping = overlapping.with_columns(
            (pl.col("chr") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8) + ":*").alias("_site_key")
        )
    return overlapping.select(["_site_key", "feature_name"])


def _empty_anndata(adata: "ad.AnnData") -> "ad.AnnData":
    """Return an empty AnnData with same obs."""
    return ad.AnnData(
        X=np.zeros((adata.n_obs, 0), dtype=np.float32),
        obs=adata.obs.copy(),
        var=pd.DataFrame(columns=["chr", "start", "end"]),
        layers={"coverage": np.zeros((adata.n_obs, 0), dtype=np.int32),
                "methylated_counts": np.zeros((adata.n_obs, 0), dtype=np.int32)},
    )

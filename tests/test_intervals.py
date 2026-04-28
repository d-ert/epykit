"""
tests/test_intervals.py
=======================
Unit tests for epykit.intervals (tile_counts, annotate_features,
annotate_cpg_islands).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_adata(sites: list[tuple[str, int, int]], n_samples: int = 3) -> ad.AnnData:
    """Build a minimal AnnData with coverage + methylated_counts layers."""
    n_sites = len(sites)
    rng = np.random.default_rng(42)
    cov = rng.integers(5, 30, size=(n_samples, n_sites), dtype=np.int32)
    meth = (cov * rng.uniform(0.1, 0.9, size=(n_samples, n_sites))).astype(np.int32)
    beta = np.where(cov > 0, meth / cov * 100.0, np.nan).astype(np.float32)

    var = pd.DataFrame(
        {"chr": [s[0] for s in sites],
         "start": [s[1] for s in sites],
         "end": [s[2] for s in sites]},
        index=[f"{s[0]}:{s[1]}-{s[2]}:*" for s in sites],
    )
    var.index.name = "locus_key"

    obs = pd.DataFrame(
        {"group": ["ctrl", "ctrl", "treat"][:n_samples]},
        index=[f"sample_{i}" for i in range(n_samples)],
    )

    return ad.AnnData(
        X=beta,
        obs=obs,
        var=var,
        layers={"coverage": cov, "methylated_counts": meth},
    )


@pytest.fixture
def small_adata() -> ad.AnnData:
    """4-site AnnData spanning chr1 and chr2."""
    return _make_adata([
        ("chr1", 1000, 1001),
        ("chr1", 2000, 2001),
        ("chr1", 3000, 3001),
        ("chr2",  500,  501),
    ])


@pytest.fixture
def features_bed(tmp_path: Path) -> Path:
    """Write a small BED file with two features."""
    bed = tmp_path / "features.bed"
    bed.write_text(
        "chr1\t900\t3500\tgene_A\n"
        "chr2\t400\t600\tgene_B\n"
    )
    return bed


@pytest.fixture
def cpgi_bed(tmp_path: Path) -> Path:
    """Write a small CpG-island BED file."""
    bed = tmp_path / "cpgi.bed"
    bed.write_text(
        "chr1\t950\t3200\tCGI_1\n"
        "chr2\t450\t550\tCGI_2\n"
    )
    return bed


# ---------------------------------------------------------------------------
# tile_counts
# ---------------------------------------------------------------------------

class TestTileCounts:
    def test_returns_anndata(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        assert isinstance(result, ad.AnnData)

    def test_has_required_var_columns(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        for col in ("chr", "start", "end"):
            assert col in result.var.columns, f"Missing var column: {col}"

    def test_obs_preserved(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        assert result.n_obs == small_adata.n_obs

    def test_tile_window_size(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        window = 2000
        result = tile_counts(small_adata, window=window)
        spans = result.var["end"] - result.var["start"]
        assert all(s <= window for s in spans), (
            f"Some tiles exceed window={window}: {spans.tolist()}"
        )

    def test_has_coverage_layer(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        assert "coverage" in result.layers

    def test_has_methylated_counts_layer(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        assert "methylated_counts" in result.layers

    def test_n_cpgs_column(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        assert "n_cpgs" in result.var.columns

    def test_smaller_window_more_tiles(self, small_adata):
        from epykit.intervals.tiling import tile_counts
        small_w = tile_counts(small_adata, window=1000)
        large_w = tile_counts(small_adata, window=5000)
        assert small_w.n_vars >= large_w.n_vars

    def test_coverage_sum_conserved(self, small_adata):
        """Total coverage in tiles must equal total coverage in original."""
        from epykit.intervals.tiling import tile_counts
        result = tile_counts(small_adata, window=5000)
        orig_total = np.asarray(small_adata.layers["coverage"]).sum()
        tile_total = np.asarray(result.layers["coverage"]).sum()
        assert tile_total == orig_total


# ---------------------------------------------------------------------------
# annotate_features
# ---------------------------------------------------------------------------

class TestAnnotateFeatures:
    def test_returns_anndata(self, small_adata, features_bed):
        from epykit.intervals.tiling import annotate_features
        result = annotate_features(small_adata, features_bed)
        assert isinstance(result, ad.AnnData)

    def test_n_vars_preserved(self, small_adata, features_bed):
        from epykit.intervals.tiling import annotate_features
        result = annotate_features(small_adata, features_bed)
        assert result.n_vars == small_adata.n_vars

    def test_feature_column_added(self, small_adata, features_bed):
        from epykit.intervals.tiling import annotate_features
        result = annotate_features(small_adata, features_bed, feature_col="feature")
        assert "feature" in result.var.columns

    def test_known_overlap(self, small_adata, features_bed):
        """chr1:1000 is inside gene_A (chr1:900-3500)."""
        from epykit.intervals.tiling import annotate_features
        result = annotate_features(small_adata, features_bed, feature_col="feature")
        # site chr1:1000-1001:* should have a non-None feature
        row = result.var[result.var["start"] == 1000]
        assert len(row) == 1
        assert row.iloc[0]["feature"] is not None, (
            "Expected chr1:1000 to overlap gene_A but feature is None"
        )

    def test_no_overlap_is_none(self, tmp_path, small_adata):
        """Sites not overlapping any feature should be None."""
        from epykit.intervals.tiling import annotate_features
        # BED far from any site
        bed = tmp_path / "far.bed"
        bed.write_text("chrX\t999000\t999999\tgene_X\n")
        result = annotate_features(small_adata, bed, feature_col="feature")
        assert all(v is None for v in result.var["feature"]), (
            "Expected all features to be None for non-overlapping BED"
        )

    def test_inplace_modifies_adata(self, small_adata, features_bed):
        from epykit.intervals.tiling import annotate_features
        result = annotate_features(small_adata, features_bed, inplace=True)
        assert result is small_adata
        assert "feature" in small_adata.var.columns

    def test_file_not_found_raises(self, small_adata, tmp_path):
        from epykit.intervals.tiling import annotate_features
        with pytest.raises(FileNotFoundError):
            annotate_features(small_adata, tmp_path / "missing.bed")


# ---------------------------------------------------------------------------
# annotate_cpg_islands
# ---------------------------------------------------------------------------

class TestAnnotateCpGIslands:
    def test_returns_anndata(self, small_adata, cpgi_bed):
        from epykit.intervals.tiling import annotate_cpg_islands
        result = annotate_cpg_islands(small_adata, cpgi_bed)
        assert isinstance(result, ad.AnnData)

    def test_n_vars_preserved(self, small_adata, cpgi_bed):
        from epykit.intervals.tiling import annotate_cpg_islands
        result = annotate_cpg_islands(small_adata, cpgi_bed)
        assert result.n_vars == small_adata.n_vars

    def test_region_column_added(self, small_adata, cpgi_bed):
        from epykit.intervals.tiling import annotate_cpg_islands
        result = annotate_cpg_islands(small_adata, cpgi_bed)
        assert "cpg_region" in result.var.columns

    def test_valid_region_values(self, small_adata, cpgi_bed):
        from epykit.intervals.tiling import annotate_cpg_islands
        result = annotate_cpg_islands(small_adata, cpgi_bed)
        valid = {"island", "shore", "shelf", "open_sea"}
        for v in result.var["cpg_region"]:
            assert v in valid, f"Unexpected region value: {v!r}"

    def test_known_island_site(self, small_adata, cpgi_bed):
        """chr1:1000 is inside CGI_1 (chr1:950-3200) → island."""
        from epykit.intervals.tiling import annotate_cpg_islands
        result = annotate_cpg_islands(small_adata, cpgi_bed)
        row = result.var[result.var["start"] == 1000]
        assert row.iloc[0]["cpg_region"] == "island", (
            f"Expected 'island' for chr1:1000, got {row.iloc[0]['cpg_region']!r}"
        )

    def test_open_sea_site(self, small_adata, cpgi_bed):
        """chr1:11000 doesn't overlap any CGI → open_sea (if present)."""
        # Add a distant site
        extra = _make_adata([
            ("chr1", 1000, 1001),
            ("chr1", 11000, 11001),  # far from CGI_1 (ends at 3200+shelf)
        ])
        from epykit.intervals.tiling import annotate_cpg_islands
        result = annotate_cpg_islands(extra, cpgi_bed)
        row = result.var[result.var["start"] == 11000]
        # with default shore=2000, shelf=4000: 3200+4000=7200 < 11000 → open_sea
        assert row.iloc[0]["cpg_region"] == "open_sea", (
            f"Expected 'open_sea' for chr1:11000, got {row.iloc[0]['cpg_region']!r}"
        )

    def test_file_not_found_raises(self, small_adata, tmp_path):
        from epykit.intervals.tiling import annotate_cpg_islands
        with pytest.raises(FileNotFoundError):
            annotate_cpg_islands(small_adata, tmp_path / "missing.bed")

    def test_polars_bio_routing(self, small_adata, cpgi_bed, monkeypatch):
        """When polars-bio is available, overlap should route via it."""
        import epykit.intervals.tiling as tiling
        from epykit.intervals.tiling import annotate_cpg_islands

        calls = {"bio": 0, "pure": 0}
        original_pure = tiling._overlap_pure_polars

        def _bio(*args, **kwargs):
            calls["bio"] += 1
            return original_pure(*args, **kwargs)

        def _pure(*args, **kwargs):
            calls["pure"] += 1
            return original_pure(*args, **kwargs)

        monkeypatch.setattr(tiling, "_HAS_POLARS_BIO", True)
        monkeypatch.setattr(tiling, "_overlap_polars_bio", _bio)
        monkeypatch.setattr(tiling, "_overlap_pure_polars", _pure)

        annotate_cpg_islands(small_adata, cpgi_bed)
        assert calls["bio"] > 0

"""
tests/test_io.py
================
Unit tests for the epykit.io data ingestion layer.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest


class TestReadBismarkCoverage:
    """Tests for io.bismark.read_bismark_coverage()."""

    def test_basic_read(self, bismark_cov_file):
        """Read a valid Bismark coverage file."""
        from epykit.io import read_bismark_coverage

        df = read_bismark_coverage(bismark_cov_file)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert "chr" in df.columns
        assert "start" in df.columns
        assert "end" in df.columns
        assert "beta" in df.columns
        assert "methylated" in df.columns
        assert "unmethylated" in df.columns
        assert "coverage" in df.columns

    def test_min_coverage_filter(self, bismark_cov_file):
        """min_coverage filter should remove low-coverage sites."""
        from epykit.io import read_bismark_coverage

        df_all = read_bismark_coverage(bismark_cov_file, min_coverage=1)
        df_filtered = read_bismark_coverage(bismark_cov_file, min_coverage=30)

        assert len(df_filtered) < len(df_all)
        assert (df_filtered["coverage"] >= 30).all()

    def test_max_coverage_filter(self, bismark_cov_file):
        """max_coverage filter should cap at upper limit."""
        from epykit.io import read_bismark_coverage

        df = read_bismark_coverage(bismark_cov_file, min_coverage=1, max_coverage=50)
        assert (df["coverage"] <= 50).all()

    def test_coverage_computed_correctly(self, bismark_cov_file):
        """Coverage = methylated + unmethylated."""
        from epykit.io import read_bismark_coverage

        df = read_bismark_coverage(bismark_cov_file)
        expected = df["methylated"] + df["unmethylated"]
        assert (df["coverage"] == expected).all()

    def test_file_not_found_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        from epykit.io import read_bismark_coverage

        with pytest.raises(FileNotFoundError):
            read_bismark_coverage("/nonexistent/path.bismark.cov")

    def test_gzip_file(self, tmp_path):
        """Gzip-compressed files should be handled transparently."""
        import gzip
        from epykit.io import read_bismark_coverage

        content = b"chr1\t1000\t1001\t50.0\t25\t25\nchr1\t2000\t2001\t80.0\t40\t10\n"
        gz_file = tmp_path / "sample.bismark.cov.gz"
        with gzip.open(gz_file, "wb") as f:
            f.write(content)

        df = read_bismark_coverage(gz_file)
        assert len(df) == 2

    def test_beta_values_in_range(self, bismark_cov_file):
        """Beta values should be between 0 and 100."""
        from epykit.io import read_bismark_coverage

        df = read_bismark_coverage(bismark_cov_file)
        assert (df["beta"] >= 0).all()
        assert (df["beta"] <= 100).all()

    def test_dtypes(self, bismark_cov_file):
        """Check column dtypes match expected schema."""
        from epykit.io import read_bismark_coverage

        df = read_bismark_coverage(bismark_cov_file)
        assert df["chr"].dtype == pl.Utf8
        assert df["start"].dtype == pl.Int32
        assert df["beta"].dtype == pl.Float64
        assert df["methylated"].dtype == pl.Int32


class TestBuildAnnData:
    """Tests for io.anndata_builder.build_anndata()."""

    def test_basic_build(self, toy_adata):
        """AnnData should have correct shape and layers."""
        import anndata as ad

        assert isinstance(toy_adata, ad.AnnData)
        assert toy_adata.n_obs == 4
        assert toy_adata.n_vars == 10
        assert "coverage" in toy_adata.layers
        assert "methylated_counts" in toy_adata.layers

    def test_obs_names(self, toy_adata):
        """Sample IDs should be in obs_names."""
        assert "ctrl_1" in toy_adata.obs_names
        assert "treat_1" in toy_adata.obs_names

    def test_var_columns(self, toy_adata):
        """var DataFrame should have chr/start/end/strand/context columns."""
        for col in ("chr", "start", "end", "strand", "context"):
            assert col in toy_adata.var.columns

    def test_x_shape(self, toy_adata):
        """X matrix shape must be (n_samples, n_sites)."""
        import numpy as np

        assert toy_adata.X.shape == (4, 10)

    def test_layers_shape(self, toy_adata):
        """All layers must have same shape as X."""
        assert toy_adata.layers["coverage"].shape == toy_adata.X.shape
        assert toy_adata.layers["methylated_counts"].shape == toy_adata.X.shape


class TestReadSamples:
    """Integration tests for io.sample_sheet.read_samples()."""

    def test_two_samples(self, sample_sheet_dir):
        """Should produce AnnData with 2 samples."""
        import anndata as ad
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata = read_samples(sheet, min_coverage=1)

        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == 2
        assert adata.n_vars > 0

    def test_obs_metadata_present(self, sample_sheet_dir):
        """Sample sheet metadata columns should appear in adata.obs."""
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata = read_samples(sheet)

        assert "group" in adata.obs.columns
        assert "batch" in adata.obs.columns

    def test_sample_sheet_not_found_raises(self):
        """Non-existent sample sheet raises FileNotFoundError."""
        from epykit.io import read_samples

        with pytest.raises(FileNotFoundError):
            read_samples("/nonexistent/sheet.csv")

    def test_duckdb_engine_basic(self, sample_sheet_dir):
        """DuckDB engine should produce valid AnnData."""
        import anndata as ad
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata = read_samples(sheet, min_coverage=1, engine="duckdb")

        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == 2
        assert adata.n_vars > 0
        assert "coverage" in adata.layers
        assert "methylated_counts" in adata.layers

    def test_duckdb_engine_does_not_polars_preload(self, sample_sheet_dir, monkeypatch):
        """DuckDB engine should not call the Polars loader (_load_single_sample)."""
        from epykit.io import read_samples
        import epykit.io.sample_sheet as ss_mod

        def _boom(*args, **kwargs):  # pragma: no cover
            raise AssertionError("_load_single_sample() should not be called for engine='duckdb'")

        monkeypatch.setattr(ss_mod, "_load_single_sample", _boom)

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata = read_samples(sheet, min_coverage=1, engine="duckdb")
        assert adata.n_obs == 2
        assert adata.n_vars > 0

    def test_duckdb_engine_var_names_int64(self, sample_sheet_dir):
        """DuckDB engine should have locus_id stored as int64 column."""
        from epykit.io import read_samples
        import numpy as np

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata = read_samples(sheet, min_coverage=1, engine="duckdb")

        # AnnData coerces var_names to strings, so check:
        # 1. var_names are numeric strings
        assert adata.var_names.dtype == object
        assert all(s.isdigit() for s in adata.var_names)
        
        # 2. locus_id column exists as int64 for genomic encoding
        assert "locus_id" in adata.var.columns
        assert adata.var["locus_id"].dtype in (np.int64, np.dtype('int64'))
        
        # 3. locus_id encodes chr and start position
        # (can decode back: chr_id = locus_id // SCALE, start = locus_id % SCALE)
        assert all(adata.var["locus_id"] >= 0)
        assert len(adata.var["locus_id"]) == adata.n_vars

        # 4. var index name does not collide with the locus_id column name
        assert adata.var.index.name != "locus_id"

    def test_duckdb_vs_polars_consistency(self, sample_sheet_dir):
        """DuckDB and Polars engines should produce same shape."""
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata_polars = read_samples(sheet, min_coverage=1, engine="polars")
        adata_duckdb = read_samples(sheet, min_coverage=1, engine="duckdb")

        # Shapes should match (same loci found)
        assert adata_polars.n_obs == adata_duckdb.n_obs
        assert adata_polars.n_vars == adata_duckdb.n_vars

    def test_duckdb_engine_zarr_roundtrip(self, sample_sheet_dir, tmp_path):
        """DuckDB + output='zarr' should write and reload successfully."""
        import anndata as ad
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        out_dir = tmp_path / "cohort.zarr"

        adata = read_samples(
            sheet,
            min_coverage=1,
            engine="duckdb",
            output="zarr",
            out_path=out_dir,
        )

        assert out_dir.exists()
        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == 2
        assert adata.n_vars > 0

    def test_zarr_var_index_collision_renamed(self, sample_sheet_dir, tmp_path):
        """Var index name collisions should be renamed before Zarr writes."""
        import pandas as pd
        from epykit.io import read_samples
        import epykit.io.sample_sheet as ss_mod

        sheet = sample_sheet_dir / "sample_sheet.csv"
        adata = read_samples(sheet, min_coverage=1)
        adata.var.index = adata.var.index.set_names("locus_key")
        adata.var["locus_key"] = pd.Series(adata.var.index, index=adata.var.index)

        ss_mod._ensure_var_index_safe(adata)
        assert adata.var.index.name == "locus_key_index"


class TestRegionsBed:
    """Tests for regions_bed filtering across engines."""

    def test_regions_bed_reduces_vars_polars(self, sample_sheet_dir, tmp_path):
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        regions = tmp_path / "regions.bed"
        regions.write_text("chr1\t0\t3000\n")

        adata_all = read_samples(sheet, min_coverage=1, engine="polars")
        adata_filtered = read_samples(
            sheet, min_coverage=1, engine="polars", regions_bed=regions
        )

        assert adata_filtered.n_vars < adata_all.n_vars

    def test_regions_bed_reduces_vars_duckdb(self, sample_sheet_dir, tmp_path):
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        regions = tmp_path / "regions.bed"
        regions.write_text("chr1\t0\t3000\n")

        adata_all = read_samples(sheet, min_coverage=1, engine="duckdb")
        adata_filtered = read_samples(
            sheet, min_coverage=1, engine="duckdb", regions_bed=regions
        )

        assert adata_filtered.n_vars < adata_all.n_vars

    def test_regions_bed_consistent_across_engines(self, sample_sheet_dir, tmp_path):
        from epykit.io import read_samples

        sheet = sample_sheet_dir / "sample_sheet.csv"
        regions = tmp_path / "regions.bed"
        regions.write_text("chr1\t0\t3000\n")

        adata_polars = read_samples(
            sheet, min_coverage=1, engine="polars", regions_bed=regions
        )
        adata_duckdb = read_samples(
            sheet, min_coverage=1, engine="duckdb", regions_bed=regions
        )

        assert adata_polars.n_vars == adata_duckdb.n_vars


class TestCxReport:
    """Coordinate consistency for CX_report (BED-style)."""

    def test_cx_report_coordinates(self, tmp_path):
        from epykit.io import read_bismark_cx_report

        content = "chr1\t10\t+\t5\t5\tCpG\tAAA\n"
        cx_file = tmp_path / "sample.CX_report.txt"
        cx_file.write_text(content)

        df = read_bismark_cx_report(cx_file, context="CpG")
        assert df["start"].item() == 9
        assert df["end"].item() == 10


class TestSavePersistence:
    """Tests for save() and load() helpers."""

    def test_save_load_h5ad(self, toy_adata, tmp_path):
        """Round-trip through HDF5 should preserve data."""
        import numpy as np
        from epykit.io import load, save

        out_path = tmp_path / "test.h5ad"
        save(toy_adata, out_path, format="h5ad")

        assert out_path.exists()
        loaded = load(out_path)

        assert loaded.n_obs == toy_adata.n_obs
        assert loaded.n_vars == toy_adata.n_vars
        np.testing.assert_allclose(
            np.asarray(loaded.X),
            np.asarray(toy_adata.X),
            rtol=1e-5,
        )

    def test_save_load_zarr(self, toy_adata, tmp_path):
        """Round-trip through Zarr should preserve data."""
        import numpy as np
        from epykit.io import load, save

        out_path = tmp_path / "test.zarr"
        save(toy_adata, out_path, format="zarr")

        assert out_path.exists()
        loaded = load(out_path)

        assert loaded.n_obs == toy_adata.n_obs
        np.testing.assert_allclose(
            np.asarray(loaded.X),
            np.asarray(toy_adata.X),
            rtol=1e-5,
        )

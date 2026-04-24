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
        assert df["start"].dtype == pl.Int64
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

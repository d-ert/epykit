"""
tests/test_core.py
==================
Unit tests for epykit.core.MethylData.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestMethylDataProperties:
    """Test MethylData accessors."""

    def test_repr(self, toy_mdata):
        """repr should contain sample count and site count."""
        r = repr(toy_mdata)
        assert "4 samples" in r
        assert "10 sites" in r

    def test_beta_shape(self, toy_mdata):
        """beta property should return (n_samples, n_sites) array."""
        assert toy_mdata.beta.shape == (4, 10)

    def test_coverage_shape(self, toy_mdata):
        """coverage property shape check."""
        assert toy_mdata.coverage.shape == (4, 10)

    def test_methylated_shape(self, toy_mdata):
        """methylated property shape check."""
        assert toy_mdata.methylated.shape == (4, 10)

    def test_unmethylated_derived(self, toy_mdata):
        """unmethylated = coverage - methylated."""
        np.testing.assert_array_equal(
            toy_mdata.unmethylated,
            toy_mdata.coverage - toy_mdata.methylated,
        )

    def test_n_samples(self, toy_mdata):
        assert toy_mdata.n_samples == 4

    def test_n_sites(self, toy_mdata):
        assert toy_mdata.n_sites == 10

    def test_sites_columns(self, toy_mdata):
        """sites should contain chr, start, end columns."""
        for col in ("chr", "start", "end"):
            assert col in toy_mdata.sites.columns

    def test_samples_has_group(self, toy_mdata):
        """samples should contain group column from fixture."""
        assert "group" in toy_mdata.samples.columns


class TestFilterCoverage:
    """Test filter_coverage method."""

    def test_all_pass(self, toy_mdata):
        """No sites should be removed when min_cov=0."""
        filtered = toy_mdata.filter_coverage(min_cov=0)
        assert filtered.n_sites == toy_mdata.n_sites

    def test_high_threshold_removes_sites(self, toy_mdata):
        """A very high min_cov should remove sites."""
        filtered = toy_mdata.filter_coverage(min_cov=1000)
        assert filtered.n_sites == 0

    def test_max_cov_filter(self, toy_mdata):
        """max_cov filter should remove high-coverage sites."""
        filtered = toy_mdata.filter_coverage(min_cov=0, max_cov=5)
        # Coverage in fixture is 10-50, so all sites should be removed
        assert filtered.n_sites == 0

    def test_returns_new_object(self, toy_mdata):
        """filter_coverage should return a new MethylData, not mutate."""
        original_n = toy_mdata.n_sites
        filtered = toy_mdata.filter_coverage(min_cov=20)
        assert toy_mdata.n_sites == original_n  # original unchanged

    def test_filtered_coverage_passes_threshold(self, toy_mdata):
        """Lenient mode keeps sites where any sample meets the threshold."""
        min_cov = 15
        filtered = toy_mdata.filter_coverage(min_cov=min_cov)
        if filtered.n_sites > 0:
            assert (filtered.coverage >= min_cov).any(axis=0).all()

    def test_require_all_samples_strictness(self, toy_mdata):
        """Strict filtering should be a subset of lenient filtering."""
        min_cov = 45
        strict = toy_mdata.filter_coverage(min_cov=min_cov, require_all_samples=True)
        lenient = toy_mdata.filter_coverage(min_cov=min_cov, require_all_samples=False)
        assert strict.n_sites <= lenient.n_sites
        assert set(strict.var_names).issubset(set(lenient.var_names))

    def test_require_all_samples_differs(self, toy_adata):
        """Strict vs lenient should differ when only some samples pass min_cov."""
        from epykit.core import MethylData

        adata = toy_adata.copy()
        adata.layers["coverage"][0, 0] = 50
        adata.layers["coverage"][1:, 0] = 1
        mdata = MethylData(adata)

        strict = mdata.filter_coverage(min_cov=10, require_all_samples=True)
        lenient = mdata.filter_coverage(min_cov=10, require_all_samples=False)
        assert strict.n_sites < lenient.n_sites


class TestSubsetContext:
    """Test subset_context method."""

    def test_cpg_subset(self, toy_mdata):
        """CpG subset should return all sites (all are CpG in fixture)."""
        cpg = toy_mdata.subset_context("CpG")
        assert cpg.n_sites == toy_mdata.n_sites

    def test_chg_empty(self, toy_mdata):
        """CHG subset should return 0 sites (none in fixture)."""
        chg = toy_mdata.subset_context("CHG")
        assert chg.n_sites == 0

    def test_invalid_context_raises(self, toy_mdata):
        """Invalid context string should raise ValueError."""
        with pytest.raises(ValueError, match="context must be one of"):
            toy_mdata.subset_context("INVALID")

    def test_returns_new_object(self, toy_mdata):
        """Returns new MethylData."""
        original_n = toy_mdata.n_sites
        sub = toy_mdata.subset_context("CpG")
        assert toy_mdata.n_sites == original_n


class TestUnite:
    """Test unite method."""

    def test_intersect_all_covered(self, toy_mdata):
        """All sites covered in all samples should all pass."""
        united = toy_mdata.unite(type="intersect")
        # All coverage values >= 10 in fixture
        assert united.n_sites == toy_mdata.n_sites

    def test_union_returns_all(self, toy_mdata):
        """Union type should return all sites."""
        united = toy_mdata.unite(type="union")
        assert united.n_sites == toy_mdata.n_sites

    def test_unite_removes_uncovered_sites(self, toy_adata):
        """Sites with 0 coverage in any sample should be removed."""
        from epykit.core import MethylData

        adata_copy = toy_adata.copy()
        # Zero out coverage for first sample at first site
        adata_copy.layers["coverage"][0, 0] = 0
        mdata = MethylData(adata_copy)
        united = mdata.unite(type="intersect")
        assert united.n_sites == toy_adata.n_vars - 1


class TestCoverageStats:
    """Test coverage_stats() summary method."""

    def test_output_columns(self, toy_mdata):
        """coverage_stats should return expected columns."""
        stats = toy_mdata.coverage_stats()
        for col in ("mean_cov", "median_cov", "n_sites", "n_sites_covered"):
            assert col in stats.columns

    def test_output_index(self, toy_mdata):
        """Index should match sample IDs."""
        stats = toy_mdata.coverage_stats()
        assert set(stats.index) == set(toy_mdata.obs_names)


class TestGlobalMethylation:
    """Test global_methylation() summary."""

    def test_output_columns(self, toy_mdata):
        """global_methylation should return expected columns."""
        gm = toy_mdata.global_methylation()
        assert "global_beta_mean" in gm.columns
        assert "global_pct_meth" in gm.columns

    def test_values_in_range(self, toy_mdata):
        """Global methylation should be between 0 and 100."""
        gm = toy_mdata.global_methylation()
        valid = gm["global_beta_mean"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestMethylDataValidation:
    """Test that invalid AnnData raises ValueError."""

    def test_missing_layers_raises(self, toy_adata):
        """AnnData without required layers should raise ValueError."""
        from epykit.core import MethylData

        bad_adata = toy_adata.copy()
        del bad_adata.layers["coverage"]

        with pytest.raises(ValueError, match="missing required layers"):
            MethylData(bad_adata)

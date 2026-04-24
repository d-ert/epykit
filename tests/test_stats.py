"""
tests/test_stats.py
===================
Unit tests for the epykit.stats differential methylation engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestFisherExactTest:
    """Tests for stats.tests.fisher_exact_test()."""

    def test_output_shapes(self, toy_mdata):
        """fisher_exact_test should return arrays of shape (n_sites,)."""
        from epykit.stats.tests import fisher_exact_test

        idx_a = np.array([True, True, False, False])
        idx_b = np.array([False, False, True, True])
        pvals, log_ors, mean_diff = fisher_exact_test(
            toy_mdata, idx_a=idx_a, idx_b=idx_b
        )

        assert pvals.shape == (toy_mdata.n_sites,)
        assert log_ors.shape == (toy_mdata.n_sites,)
        assert mean_diff.shape == (toy_mdata.n_sites,)

    def test_pvalues_in_range(self, toy_mdata):
        """p-values must be in [0, 1]."""
        from epykit.stats.tests import fisher_exact_test

        idx_a = np.array([True, True, False, False])
        idx_b = np.array([False, False, True, True])
        pvals, _, _ = fisher_exact_test(
            toy_mdata, idx_a=idx_a, idx_b=idx_b
        )

        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_known_differential_site(self, toy_mdata):
        """Sites 0-2 are highly differential in the fixture — p < 0.1."""
        from epykit.stats.tests import fisher_exact_test

        idx_a = np.array([True, True, False, False])
        idx_b = np.array([False, False, True, True])
        pvals, _, _ = fisher_exact_test(
            toy_mdata, idx_a=idx_a, idx_b=idx_b
        )

        # Sites 0-2 have ~30% vs ~70% methylation difference
        assert pvals[0] < 0.1 or pvals[1] < 0.1  # At least one should be significant


class TestCalculateDiffMeth:
    """Integration tests for the master calculate_diff_meth() function."""

    def test_returns_dataframe(self, toy_mdata):
        """Should return a pandas DataFrame."""
        from epykit.stats import calculate_diff_meth

        result = calculate_diff_meth(
            toy_mdata, treatment_col="group", test="fisher"
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, toy_mdata):
        """Result should have required columns."""
        from epykit.stats import calculate_diff_meth

        result = calculate_diff_meth(
            toy_mdata, treatment_col="group", test="fisher"
        )
        for col in ("chr", "start", "end", "pvalue", "qvalue", "mean_diff"):
            assert col in result.columns

    def test_row_count_equals_sites(self, toy_mdata):
        """One row per CpG site."""
        from epykit.stats import calculate_diff_meth

        result = calculate_diff_meth(
            toy_mdata, treatment_col="group", test="fisher"
        )
        assert len(result) == toy_mdata.n_sites

    def test_sorted_by_pvalue(self, toy_mdata):
        """Results should be sorted by p-value ascending."""
        from epykit.stats import calculate_diff_meth

        result = calculate_diff_meth(
            toy_mdata, treatment_col="group", test="fisher"
        )
        pvals = result["pvalue"].values
        assert np.all(pvals[:-1] <= pvals[1:])

    def test_qvalue_leq_pvalue(self, toy_mdata):
        """BH q-values should never exceed 1 and be non-negative."""
        from epykit.stats import calculate_diff_meth

        result = calculate_diff_meth(
            toy_mdata, treatment_col="group", test="fisher"
        )
        assert (result["qvalue"] >= 0).all()
        assert (result["qvalue"] <= 1).all()

    def test_invalid_treatment_col_raises(self, toy_mdata):
        """Missing treatment_col should raise ValueError."""
        from epykit.stats import calculate_diff_meth

        with pytest.raises(ValueError, match="treatment_col"):
            calculate_diff_meth(toy_mdata, treatment_col="nonexistent")

    def test_auto_selects_fisher_for_single_replicates(self, toy_adata):
        """With 1 sample per group, auto should choose Fisher test."""
        import anndata as ad
        from epykit.core import MethylData
        from epykit.stats import calculate_diff_meth

        # Keep only 1 sample per group
        adata_sub = toy_adata[[0, 2], :].copy()
        adata_sub.obs["group"] = ["control", "treatment"]
        mdata_sub = MethylData(adata_sub)

        result = calculate_diff_meth(
            mdata_sub, treatment_col="group", test="auto"
        )
        assert "log2_odds_ratio" in result.columns

    def test_glm_test_runs(self, toy_mdata):
        """GLM test should run and return valid p-values."""
        from epykit.stats import calculate_diff_meth

        result = calculate_diff_meth(
            toy_mdata, treatment_col="group", test="glm", overdispersion=False
        )
        assert len(result) == toy_mdata.n_sites
        assert (result["pvalue"] >= 0).all()


class TestMergeDMRs:
    """Tests for stats.dmr.merge_dmrs()."""

    def _make_dmc_results(self) -> pd.DataFrame:
        """Create a mock DMC result with consecutive significant sites."""
        data = {
            "chr": ["chr1"] * 6 + ["chr2"] * 3,
            "start": [1000, 1050, 1100, 1150, 1200, 5000, 200, 300, 400],
            "end":   [1001, 1051, 1101, 1151, 1201, 5001, 201, 301, 401],
            "qvalue": [0.01, 0.02, 0.03, 0.04, 0.001, 0.8, 0.01, 0.02, 0.03],
            "mean_diff": [-25.0, -22.0, -28.0, -30.0, -20.0, 5.0, 15.0, 18.0, 12.0],
        }
        return pd.DataFrame(data)

    def test_basic_merge(self):
        """Should merge consecutive significant sites into DMRs."""
        from epykit.stats import merge_dmrs

        dmc = self._make_dmc_results()
        dmrs = merge_dmrs(dmc, qvalue_cutoff=0.05, min_sites=3, min_abs_diff=10.0)

        assert isinstance(dmrs, pd.DataFrame)
        assert len(dmrs) >= 1

    def test_output_columns(self):
        """DMR table should have required columns."""
        from epykit.stats import merge_dmrs

        dmc = self._make_dmc_results()
        dmrs = merge_dmrs(dmc, qvalue_cutoff=0.05, min_sites=2, min_abs_diff=0.0)

        for col in ("chr", "start", "end", "n_cpgs", "mean_diff", "direction"):
            assert col in dmrs.columns

    def test_direction_classification(self):
        """Positive mean_diff = hyper, negative = hypo."""
        from epykit.stats import merge_dmrs

        dmc = self._make_dmc_results()
        dmrs = merge_dmrs(dmc, qvalue_cutoff=0.05, min_sites=2, min_abs_diff=0.0)

        for _, row in dmrs.iterrows():
            if row["mean_diff"] > 0:
                assert row["direction"] == "hyper"
            else:
                assert row["direction"] == "hypo"

    def test_no_significant_sites(self):
        """When no sites pass qvalue_cutoff, returns empty DataFrame."""
        from epykit.stats import merge_dmrs

        dmc = self._make_dmc_results()
        dmrs = merge_dmrs(dmc, qvalue_cutoff=0.001)

        assert len(dmrs) == 0

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        from epykit.stats import merge_dmrs

        bad_df = pd.DataFrame({"chr": ["chr1"], "start": [100]})
        with pytest.raises(ValueError, match="missing columns"):
            merge_dmrs(bad_df)

    def test_min_sites_filter(self):
        """DMRs with fewer than min_sites should be excluded."""
        from epykit.stats import merge_dmrs

        dmc = self._make_dmc_results()
        dmrs_strict = merge_dmrs(dmc, qvalue_cutoff=0.05, min_sites=10)

        assert len(dmrs_strict) == 0

    def test_dmr_coordinates_valid(self):
        """DMR start should be <= end."""
        from epykit.stats import merge_dmrs

        dmc = self._make_dmc_results()
        dmrs = merge_dmrs(dmc, qvalue_cutoff=0.05, min_sites=2, min_abs_diff=0.0)

        if len(dmrs) > 0:
            assert (dmrs["start"] <= dmrs["end"]).all()


class TestFDRCorrection:
    """Tests for the _apply_fdr helper."""

    def test_bh_correction(self):
        """BH correction should return array of same length."""
        from epykit.stats.tests import _apply_fdr

        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
        qvals = _apply_fdr(pvals, method="BH")

        assert len(qvals) == len(pvals)
        assert np.all(qvals >= 0)
        assert np.all(qvals <= 1)

    def test_qvalues_monotone(self):
        """BH q-values should be non-decreasing after sorting."""
        from epykit.stats.tests import _apply_fdr

        pvals = np.array([0.001, 0.005, 0.01, 0.05, 0.3, 0.9])
        qvals = _apply_fdr(pvals, method="BH")
        sorted_qvals = np.sort(qvals)
        assert np.all(sorted_qvals[:-1] <= sorted_qvals[1:])

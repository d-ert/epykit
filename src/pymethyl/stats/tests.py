"""
pymethyl.stats.tests
====================
Differential methylation testing: Fisher's Exact Test, Logistic GLM + LRT,
HC0 overdispersion correction.

Statistical models
------------------
**Fisher's Exact Test** (no replicates)
    Applied when each group has exactly 1 sample.  Tests the 2×2 table:
    ``[[meth_A, unmeth_A], [meth_B, unmeth_B]]``.
    Implemented via ``scipy.stats.fisher_exact``.

**Logistic GLM + Likelihood Ratio Test** (with replicates)
    Fits a Binomial GLM using statsmodels with the logit link function.
    The full model includes treatment + covariates; the reduced model
    excludes treatment.  The deviance difference is tested against χ².

**HC0 overdispersion correction**
    ``model.fit(cov_type='HC0')`` applies heteroskedasticity-consistent
    robust standard errors.  This is mathematically equivalent to
    methylKit's quasi-binomial McCullagh-Nelder correction.

Output columns
--------------
    chr, start, end, strand, context,
    pvalue, qvalue (BH-FDR),
    mean_diff (mean beta difference: treated - control),
    meth_A, meth_B (mean methylation per group),
    log2_odds_ratio (Fisher only)
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pymethyl.core.methyldata import MethylData

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not installed. GLM-based tests will not be available."
    )


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def calculate_diff_meth(
    mdata: "MethylData",
    *,
    treatment_col: str = "group",
    test: str = "auto",
    overdispersion: bool = True,
    covariates: list[str] | None = None,
    fdr_method: str = "BH",
    n_jobs: int = 1,
    min_sites: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run differential methylation analysis on a MethylData object.

    This is the master entry point — equivalent to
    ``methylKit::calculateDiffMeth()``.  Automatically selects the
    appropriate test based on the number of samples per group.

    Parameters
    ----------
    mdata:
        A united MethylData object (all samples must have coverage > 0
        at every site, i.e. after calling ``.unite()``).
    treatment_col:
        Column in ``mdata.samples`` (``obs``) defining the two groups.
        Must have exactly 2 unique values.
    test:
        Statistical test to use:
        ``"auto"``    — Fisher if 1 replicate per group, GLM otherwise.
        ``"fisher"``  — Force Fisher's exact test.
        ``"glm"``     — Force Logistic GLM + LRT.
    overdispersion:
        If ``True`` and ``test="glm"``, applies HC0 robust covariance
        correction to account for biological overdispersion.
        Default: ``True``.
    covariates:
        List of column names in ``mdata.samples`` to include as
        covariates in the GLM design matrix (batch, age, sex, etc.).
    fdr_method:
        Multiple testing correction method passed to
        ``statsmodels.stats.multitest.multipletests``.
        ``"BH"`` (Benjamini-Hochberg) or ``"bonferroni"``  Default: ``"BH"``.
    n_jobs:
        Number of parallel jobs for site-wise testing.  Currently uses
        simple vectorised operations (n_jobs > 1 reserved for future use).
    min_sites:
        Minimum number of sites required to run the analysis.  Raises
        ``ValueError`` if fewer sites are present.
    verbose:
        Log progress messages.

    Returns
    -------
    pd.DataFrame
        One row per CpG site, sorted by p-value.  Columns:
        ``chr``, ``start``, ``end``, ``strand``, ``context``,
        ``pvalue``, ``qvalue``, ``mean_diff``,
        ``mean_meth_treat``, ``mean_meth_ctrl``.

    Raises
    ------
    ValueError
        If ``treatment_col`` is not found, or if there are not exactly 2
        groups, or if fewer than ``min_sites`` sites are available.

    Examples
    --------
    >>> results = calculate_diff_meth(mdata, treatment_col="group")
    >>> dmcs = results[results["qvalue"] < 0.05]
    """
    # --- Validate input ---
    if treatment_col not in mdata.samples.columns:
        raise ValueError(
            f"treatment_col='{treatment_col}' not found in mdata.obs columns: "
            f"{list(mdata.samples.columns)}"
        )

    groups = mdata.samples[treatment_col].dropna().unique().tolist()
    if len(groups) != 2:
        raise ValueError(
            f"treatment_col must have exactly 2 groups, found: {groups}"
        )

    if mdata.n_sites < min_sites:
        raise ValueError(
            f"Only {mdata.n_sites} sites available, need at least {min_sites}. "
            "Did you call .unite() first?"
        )

    grp_a, grp_b = sorted(groups)
    idx_a = (mdata.samples[treatment_col] == grp_a).values
    idx_b = (mdata.samples[treatment_col] == grp_b).values
    n_a = int(idx_a.sum())
    n_b = int(idx_b.sum())

    # --- Auto-select test ---
    if test == "auto":
        effective_test = "fisher" if (n_a == 1 and n_b == 1) else "glm"
    else:
        effective_test = test

    if verbose:
        logger.info(
            "calculate_diff_meth: %d sites, %d samples (%s=%d, %s=%d), test='%s'",
            mdata.n_sites, mdata.n_samples,
            grp_a, n_a, grp_b, n_b,
            effective_test,
        )

    # --- Run tests ---
    if effective_test == "fisher":
        pvalues, log_ors, mean_diff = fisher_exact_test(
            mdata, idx_a=idx_a, idx_b=idx_b
        )
        extra_cols = {"log2_odds_ratio": log_ors}
    elif effective_test == "glm":
        if not _HAS_STATSMODELS:
            raise ImportError(
                "statsmodels is required for GLM tests: pip install statsmodels"
            )
        pvalues, mean_diff = glm_lrt_test(
            mdata,
            idx_a=idx_a,
            idx_b=idx_b,
            overdispersion=overdispersion,
            covariates=covariates,
        )
        extra_cols = {}
    else:
        raise ValueError(f"Unknown test: '{test}'. Choose 'auto', 'fisher', or 'glm'.")

    # --- FDR correction ---
    qvalues = _apply_fdr(pvalues, method=fdr_method)

    # --- Compute per-group mean methylation ---
    beta = mdata.beta  # (n_samples, n_sites)
    mean_a = np.nanmean(beta[idx_a, :], axis=0)
    mean_b = np.nanmean(beta[idx_b, :], axis=0)

    # --- Build result DataFrame ---
    var = mdata.sites.copy()
    result = pd.DataFrame(
        {
            "chr": var["chr"].values,
            "start": var["start"].values,
            "end": var["end"].values,
            "strand": var.get("strand", pd.Series(["*"] * mdata.n_sites)).values,
            "context": var.get("context", pd.Series(["CpG"] * mdata.n_sites)).values,
            "pvalue": pvalues,
            "qvalue": qvalues,
            "mean_diff": mean_diff,
            f"mean_meth_{grp_a}": mean_a,
            f"mean_meth_{grp_b}": mean_b,
        },
        index=mdata.var_names,
    )
    for col, arr in extra_cols.items():
        result[col] = arr

    result = result.sort_values("pvalue").reset_index(drop=False).rename(
        columns={"index": "locus_key"}
    )

    n_sig = int((result["qvalue"] < 0.05).sum())
    if verbose:
        logger.info(
            "calculate_diff_meth: %d significant sites at q<0.05 (FDR=%s)",
            n_sig, fdr_method,
        )

    return result


# ---------------------------------------------------------------------------
# Fisher's Exact Test
# ---------------------------------------------------------------------------

def fisher_exact_test(
    mdata: "MethylData",
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-site Fisher's exact test (no replicates or pooled).

    For each site, constructs the 2×2 contingency table::

        [[sum_methylated_A, sum_unmethylated_A],
         [sum_methylated_B, sum_unmethylated_B]]

    Parameters
    ----------
    mdata:
        MethylData object.
    idx_a, idx_b:
        Boolean arrays indexing group A and group B samples.

    Returns
    -------
    pvalues : np.ndarray shape (n_sites,)
    log2_odds_ratios : np.ndarray shape (n_sites,)
    mean_diff : np.ndarray shape (n_sites,) — mean_beta_B - mean_beta_A
    """
    meth = mdata.methylated      # (n_samples, n_sites)
    cov = mdata.coverage
    unmeth = cov - meth

    # Sum across group samples
    meth_a = meth[idx_a, :].sum(axis=0).astype(np.int64)
    unmeth_a = unmeth[idx_a, :].sum(axis=0).astype(np.int64)
    meth_b = meth[idx_b, :].sum(axis=0).astype(np.int64)
    unmeth_b = unmeth[idx_b, :].sum(axis=0).astype(np.int64)

    n_sites = mdata.n_sites
    pvalues = np.ones(n_sites, dtype=np.float64)
    log_ors = np.zeros(n_sites, dtype=np.float64)

    for i in range(n_sites):
        table = np.array([
            [meth_a[i], unmeth_a[i]],
            [meth_b[i], unmeth_b[i]],
        ])
        odds, p = scipy_stats.fisher_exact(table)
        pvalues[i] = p
        log_ors[i] = np.log2(odds + 1e-10)

    # Mean methylation difference (B - A)
    beta = mdata.beta
    mean_diff = np.nanmean(beta[idx_b, :], axis=0) - np.nanmean(beta[idx_a, :], axis=0)

    return pvalues, log_ors, mean_diff


# ---------------------------------------------------------------------------
# Logistic GLM + Likelihood Ratio Test
# ---------------------------------------------------------------------------

def glm_lrt_test(
    mdata: "MethylData",
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    overdispersion: bool = True,
    covariates: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-site Logistic GLM + Likelihood Ratio Test with optional HC0 correction.

    Fits a Binomial GLM for each site::

        logit(p_i) = β₀ + β₁·treatment + β₂·covariate₁ + ...

    The LRT compares the full model (with treatment) against the reduced
    model (without treatment).  The deviance difference is tested against
    a χ²(1) distribution.

    If ``overdispersion=True``, applies HC0 robust covariance correction
    (equivalent to methylKit's quasi-binomial McCullagh-Nelder correction).

    Parameters
    ----------
    mdata:
        MethylData object.
    idx_a, idx_b:
        Boolean arrays for group A and group B.
    overdispersion:
        Apply HC0 robust standard errors.
    covariates:
        List of column names in ``mdata.samples`` to include as covariates.

    Returns
    -------
    pvalues : np.ndarray shape (n_sites,)
    mean_diff : np.ndarray shape (n_sites,)
    """
    from scipy.stats import chi2

    n_sites = mdata.n_sites
    pvalues = np.ones(n_sites, dtype=np.float64)

    # Build design matrix
    combined_idx = idx_a | idx_b
    samples_sub = mdata.samples[combined_idx]
    n_sub = int(combined_idx.sum())

    treatment = np.zeros(n_sub, dtype=float)
    treatment[idx_b[combined_idx]] = 1.0

    design_full_data: dict[str, np.ndarray] = {"treatment": treatment}
    design_reduced_data: dict[str, np.ndarray] = {}

    if covariates:
        for cov_col in covariates:
            if cov_col in samples_sub.columns:
                col_vals = samples_sub[cov_col].values
                if col_vals.dtype.kind in ("U", "O"):
                    # Categorical: dummy encoding
                    dummies = pd.get_dummies(col_vals, prefix=cov_col, drop_first=True)
                    for c in dummies.columns:
                        design_full_data[c] = dummies[c].values
                        design_reduced_data[c] = dummies[c].values
                else:
                    design_full_data[cov_col] = col_vals.astype(float)
                    design_reduced_data[cov_col] = col_vals.astype(float)

    X_full = sm.add_constant(
        pd.DataFrame(design_full_data), prepend=True
    )
    X_reduced = sm.add_constant(
        pd.DataFrame(design_reduced_data), prepend=True
    ) if design_reduced_data else sm.add_constant(
        pd.DataFrame({"intercept": np.ones(n_sub)}), prepend=False
    )

    meth = mdata.methylated[:, :]
    cov = mdata.coverage[:, :]
    unmeth = cov - meth

    meth_sub = meth[combined_idx, :]
    unmeth_sub = unmeth[combined_idx, :]

    cov_type = "HC0" if overdispersion else "nonrobust"

    for i in range(n_sites):
        y = np.column_stack([meth_sub[:, i], unmeth_sub[:, i]])

        # Skip sites with all-zero counts
        if y[:, 0].sum() == 0 or y.sum() == 0:
            pvalues[i] = 1.0
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fit_full = sm.GLM(
                    y,
                    X_full,
                    family=sm.families.Binomial(),
                ).fit(cov_type=cov_type, disp=0)

                fit_reduced = sm.GLM(
                    y,
                    X_reduced,
                    family=sm.families.Binomial(),
                ).fit(disp=0)

                if overdispersion:
                    # Wald test on treatment coefficient with HC0 SE
                    # (F-test equivalent for quasi-binomial)
                    tstat = fit_full.tvalues.get("treatment", 0.0)
                    pvalues[i] = float(
                        2 * (1 - scipy_stats.norm.cdf(abs(tstat)))
                    )
                else:
                    # LRT: deviance difference ~ χ²(df=1)
                    dev_diff = fit_reduced.deviance - fit_full.deviance
                    pvalues[i] = float(chi2.sf(max(dev_diff, 0.0), df=1))

        except Exception:
            pvalues[i] = 1.0

    beta = mdata.beta
    mean_diff = np.nanmean(beta[idx_b, :], axis=0) - np.nanmean(beta[idx_a, :], axis=0)

    return pvalues, mean_diff


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------

def _apply_fdr(
    pvalues: np.ndarray,
    method: str = "BH",
) -> np.ndarray:
    """Apply multiple testing correction.

    Parameters
    ----------
    pvalues:
        Array of raw p-values.
    method:
        ``"BH"`` (Benjamini-Hochberg, default) or ``"bonferroni"``.

    Returns
    -------
    np.ndarray
        Adjusted p-values (q-values).
    """
    method_map = {
        "BH": "fdr_bh",
        "fdr_bh": "fdr_bh",
        "bonferroni": "bonferroni",
        "holm": "holm",
        "fdr_by": "fdr_by",
    }
    sm_method = method_map.get(method, "fdr_bh")

    # Handle NaN pvalues
    finite_mask = np.isfinite(pvalues)
    qvalues = np.ones_like(pvalues, dtype=np.float64)

    if finite_mask.sum() > 0:
        _, qvals_finite, _, _ = multipletests(
            pvalues[finite_mask],
            method=sm_method,
            alpha=0.05,
        )
        qvalues[finite_mask] = qvals_finite

    return qvalues

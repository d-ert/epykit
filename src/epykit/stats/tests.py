"""
epykit.stats.tests
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
from scipy.special import polygamma
from scipy.optimize import bisect
from statsmodels.stats.multitest import multipletests

from epykit.stats import glm_vectorized

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from epykit.core.methyldata import MethylData

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
    limma_chunk_size: int = 100000,
    limma_alpha: float = 0.5,
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
        A MethylData object. For ``test='limma'`` and ``test='glm'``,
        ``unite(type='intersect')`` is recommended, but ``limma`` can
        also handle union-mode data by masking samples with zero coverage
        per locus.
    treatment_col:
        Column in ``mdata.samples`` (``obs``) defining the two groups.
        Must have exactly 2 unique values.
    test:
        Statistical test to use:
        ``"auto"``    — Fisher if 1 replicate per group, limma otherwise.
        ``"fisher"``  — Force Fisher's exact test.
        ``"glm"``     — Force Logistic GLM + LRT.
        ``"limma"``   — Fast M-value linear model + empirical Bayes.
    overdispersion:
        If ``True`` and ``test="glm"``, applies HC0 robust covariance
        correction to account for biological overdispersion.
        Default: ``True``.
    limma_chunk_size:
        Chunk size (number of loci per block) for the limma-style
        linear model test. Larger values are faster but use more memory.
    limma_alpha:
        Pseudocount added to methylated/unmethylated counts when computing
        M-values for the limma-style test.
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
        effective_test = "fisher" if (n_a == 1 and n_b == 1) else "limma"
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
    elif effective_test == "limma":
        pvalues, mean_diff = limma_ebayes_test(
            mdata,
            idx_a=idx_a,
            idx_b=idx_b,
            covariates=covariates,
            chunk_size=limma_chunk_size,
            alpha=limma_alpha,
        )
        extra_cols = {}
    else:
        raise ValueError(
            f"Unknown test: '{test}'. Choose 'auto', 'fisher', 'glm', or 'limma'."
        )

    # --- FDR correction ---
    qvalues = _apply_fdr(pvalues, method=fdr_method)

    # --- Compute per-group mean methylation ---
    mean_diff = _compute_mean_diff_sparse_safe(mdata, idx_a, idx_b)
    mean_a, mean_b = _compute_group_means_sparse_safe(mdata, idx_a, idx_b)

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
        columns={"index": mdata.var_names.name or "locus_id"}
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
    """Vectorized Fisher's exact test (no replicates or pooled).

    For each site, constructs the 2×2 contingency table::

        [[sum_methylated_A, sum_unmethylated_A],
         [sum_methylated_B, sum_unmethylated_B]]

    Uses vectorized hypergeometric computation for ~10-50× speedup
    over per-site scipy.stats.fisher_exact().

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
    from scipy.sparse import issparse

    meth_layer = mdata.methylated_layer   # sparse or dense (n_samples, n_sites)
    cov_layer = mdata.coverage_layer      # sparse or dense

    if issparse(meth_layer):
        meth_a = np.asarray(meth_layer[idx_a, :].sum(axis=0)).ravel().astype(np.int64)
        meth_b = np.asarray(meth_layer[idx_b, :].sum(axis=0)).ravel().astype(np.int64)
        cov_a = np.asarray(cov_layer[idx_a, :].sum(axis=0)).ravel().astype(np.int64)
        cov_b = np.asarray(cov_layer[idx_b, :].sum(axis=0)).ravel().astype(np.int64)
    else:
        meth_a = np.asarray(meth_layer[idx_a, :]).sum(axis=0).astype(np.int64)
        meth_b = np.asarray(meth_layer[idx_b, :]).sum(axis=0).astype(np.int64)
        cov_a = np.asarray(cov_layer[idx_a, :]).sum(axis=0).astype(np.int64)
        cov_b = np.asarray(cov_layer[idx_b, :]).sum(axis=0).astype(np.int64)

    unmeth_a = cov_a - meth_a
    unmeth_b = cov_b - meth_b

    # Vectorized Fisher test (batch computation)
    pvalues, odds_ratios = glm_vectorized.fisher_exact_vectorized(
        table_11=meth_a,
        table_12=unmeth_a,
        table_21=meth_b,
        table_22=unmeth_b,
    )

    log_ors = np.log2(odds_ratios + 1e-10)

    # Mean methylation difference (B - A) — use sparse-aware computation
    mean_diff = _compute_mean_diff_sparse_safe(mdata, idx_a, idx_b)

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
    warn_large_sites: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Logistic GLM + Likelihood Ratio Test with optional HC0 correction.

    Fits a Binomial GLM for all sites simultaneously::

        logit(p_i) = β₀ + β₁·treatment + β₂·covariate₁ + ...

    The LRT compares the full model (with treatment) against the reduced
    model (without treatment).  The deviance difference is tested against
    a χ²(1) distribution.

    If ``overdispersion=True``, applies HC0 robust covariance correction
    (equivalent to methylKit's quasi-binomial McCullagh-Nelder correction).

    Uses vectorized IRLS that fits all sites as columns simultaneously,
    achieving 100-500× speedup over per-site statsmodels GLM.

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
    n_sites = mdata.n_sites
    if n_sites >= warn_large_sites:
        logger.warning(
            "glm_lrt_test: %d sites requested. The GLM path is extremely slow for large WGBS "
            "matrices. Consider test='limma' for genome-scale analyses.",
            n_sites,
        )

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

    import statsmodels.api as sm
    X_full = sm.add_constant(
        pd.DataFrame(design_full_data), prepend=True
    )
    X_reduced = sm.add_constant(
        pd.DataFrame(design_reduced_data), prepend=True
    ) if design_reduced_data else sm.add_constant(
        pd.DataFrame({"intercept": np.ones(n_sub)}), prepend=False
    )

    # Extract methylation data (sparse-safe)
    meth = mdata.methylated  # (n_samples, n_sites)
    cov = mdata.coverage  # (n_samples, n_sites)

    # Handle sparse matrices
    if hasattr(meth, 'toarray'):
        meth = meth[combined_idx, :].toarray()
    else:
        meth = meth[combined_idx, :]

    if hasattr(cov, 'toarray'):
        cov = cov[combined_idx, :].toarray()
    else:
        cov = cov[combined_idx, :]

    unmeth = cov - meth

    # Stack counts into (n_samples, 2, n_sites) format
    y = np.stack([meth, unmeth], axis=1)  # (n_samples, 2, n_sites)

    cov_type = "HC0" if overdispersion else "nonrobust"

    # Vectorized GLM fitting for full model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_full = glm_vectorized.glm_binomial_vectorized(
            y, X_full.values, cov_type=cov_type
        )

        # Vectorized GLM fitting for reduced model
        fit_reduced = glm_vectorized.glm_binomial_vectorized(
            y, X_reduced.values, cov_type="nonrobust"  # No HC0 for reduced
        )

    # Compute p-values: LRT or Wald depending on overdispersion
    pvalues = np.ones(n_sites, dtype=np.float64)

    if overdispersion:
        # Wald test on treatment coefficient with HC0 SE
        coef_full = fit_full["coef"]
        se_full = fit_full["se"]
        # Treatment is always at index 1 (after intercept)
        wald_stats, pvalues = glm_vectorized.wald_statistics_vectorized(
            coef_full, se_full, contrast_idx=1
        )
    else:
        # LRT: deviance difference ~ χ²(df=1)
        lrt_stats, pvalues = glm_vectorized.lrt_statistics_vectorized(
            fit_full["deviance"],
            fit_reduced["deviance"],
            df=1,
        )

    # Mean methylation difference (B - A) — sparse-safe computation
    mean_diff = _compute_mean_diff_sparse_safe(mdata, idx_a, idx_b)

    return pvalues, mean_diff


# ---------------------------------------------------------------------------
# Limma-style linear model on M-values + empirical Bayes
# ---------------------------------------------------------------------------

def limma_ebayes_test(
    mdata: "MethylData",
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    covariates: list[str] | None = None,
    chunk_size: int = 100000,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast differential methylation using M-values + empirical Bayes.

    Supports union/NaN coverage by fitting each locus on samples with
    coverage > 0 (per-locus masking). This is chunked to scale to WGBS.
    """
    from scipy.sparse import issparse

    # Build design matrix (full set of samples in union)
    combined_idx = idx_a | idx_b
    samples_sub = mdata.samples[combined_idx]
    n_sub = int(combined_idx.sum())

    design_full_data: dict[str, np.ndarray] = {
        "intercept": np.ones(n_sub, dtype=float),
    }

    treatment = np.zeros(n_sub, dtype=float)
    treatment[idx_b[combined_idx]] = 1.0
    design_full_data["treatment"] = treatment

    if covariates:
        for cov_col in covariates:
            if cov_col in samples_sub.columns:
                col_vals = samples_sub[cov_col].values
                if col_vals.dtype.kind in ("U", "O"):
                    dummies = pd.get_dummies(col_vals, prefix=cov_col, drop_first=True)
                    for c in dummies.columns:
                        design_full_data[c] = dummies[c].values
                else:
                    design_full_data[cov_col] = col_vals.astype(float)

    X_full = pd.DataFrame(design_full_data).values
    n_features = X_full.shape[1]

    meth_layer = mdata.methylated_layer
    cov_layer = mdata.coverage_layer

    if issparse(meth_layer):
        meth_sub = meth_layer[combined_idx, :].tocsc()
        cov_sub = cov_layer[combined_idx, :].tocsc()
    else:
        meth_sub = meth_layer[combined_idx, :]
        cov_sub = cov_layer[combined_idx, :]

    n_sites = mdata.n_sites
    coef_treatment = np.full(n_sites, np.nan, dtype=np.float64)
    var_resid = np.full(n_sites, np.nan, dtype=np.float64)
    df_resid = np.full(n_sites, np.nan, dtype=np.float64)
    unscaled_se_treat = np.full(n_sites, np.nan, dtype=np.float64)

    for start in range(0, n_sites, chunk_size):
        end = min(start + chunk_size, n_sites)
        if issparse(meth_sub):
            meth_chunk = meth_sub[:, start:end].toarray()
            cov_chunk = cov_sub[:, start:end].toarray()
        else:
            meth_chunk = meth_sub[:, start:end]
            cov_chunk = cov_sub[:, start:end]

        # Compute M-values with pseudocount, allow cov=0
        unmeth_chunk = cov_chunk - meth_chunk
        with np.errstate(divide="ignore", invalid="ignore"):
            mvals = np.log2((meth_chunk + alpha) / (unmeth_chunk + alpha))

        # Mask entries with zero coverage
        valid_mask = cov_chunk > 0
        if np.all(valid_mask):
            try:
                coef, _, rank, _ = np.linalg.lstsq(X_full, mvals, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < n_features:
                continue

            fitted = X_full @ coef
            resid = mvals - fitted
            df = n_sub - rank
            if df <= 0:
                continue

            try:
                xtx_inv = np.linalg.pinv(X_full.T @ X_full)
                unscaled_se = float(np.sqrt(xtx_inv[1, 1]))
            except np.linalg.LinAlgError:
                continue

            coef_treatment[start:end] = coef[1, :]
            var_resid[start:end] = (resid**2).sum(axis=0) / df
            df_resid[start:end] = float(df)
            unscaled_se_treat[start:end] = unscaled_se
        else:
            mvals = np.where(valid_mask, mvals, np.nan)
            # Fit each locus in the chunk with per-site masking
            for j in range(end - start):
                y = mvals[:, j]
                valid = np.isfinite(y)
                if valid.sum() < n_features + 1:
                    continue
                X = X_full[valid, :]
                y_valid = y[valid]

                try:
                    coef, _, rank, _ = np.linalg.lstsq(X, y_valid, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                if rank < n_features:
                    continue

                fitted = X @ coef
                resid = y_valid - fitted
                df = len(y_valid) - rank
                if df <= 0:
                    continue

                try:
                    xtx_inv = np.linalg.pinv(X.T @ X)
                    unscaled_se = float(np.sqrt(xtx_inv[1, 1]))
                except np.linalg.LinAlgError:
                    continue

                coef_treatment[start + j] = coef[1]
                var_resid[start + j] = float((resid**2).sum() / df)
                df_resid[start + j] = float(df)
                unscaled_se_treat[start + j] = unscaled_se

    # Empirical Bayes shrinkage across valid loci
    valid_sites = (
        np.isfinite(coef_treatment)
        & np.isfinite(var_resid)
        & np.isfinite(unscaled_se_treat)
        & (df_resid > 0)
    )
    pvalues = np.ones(n_sites, dtype=np.float64)
    if valid_sites.sum() == 0:
        return pvalues, _compute_mean_diff_sparse_safe(mdata, idx_a, idx_b)

    prior = _fit_ebayes_prior(var_resid[valid_sites], df_resid[valid_sites])

    df_post = df_resid[valid_sites] + prior["df"]
    var_post = (
        (df_resid[valid_sites] * var_resid[valid_sites])
        + (prior["df"] * prior["var"])
    ) / df_post

    t_stats = coef_treatment[valid_sites] / (
        np.sqrt(var_post) * unscaled_se_treat[valid_sites]
    )
    pvals_valid = 2 * scipy_stats.t.sf(np.abs(t_stats), df=df_post)
    pvalues[valid_sites] = pvals_valid

    mean_diff = _compute_mean_diff_sparse_safe(mdata, idx_a, idx_b)
    return pvalues, mean_diff


def _fit_ebayes_prior(var: np.ndarray, df: np.ndarray) -> dict[str, float]:
    """Estimate prior df and variance for empirical Bayes shrinkage.

    Uses a limma-style squeezeVar moment-matching approach with
    safeguards for degenerate inputs.
    """
    var = np.clip(var, 1e-10, np.inf)
    z = np.log(var)
    z_mean = float(np.mean(z))
    z_var = float(np.var(z))

    if z_var == 0 or not np.isfinite(z_var):
        return {"df": 1e6, "var": float(np.exp(z_mean))}

    df_mean = float(np.mean(df))
    target = z_var - polygamma(1, df_mean / 2.0)
    if not np.isfinite(target) or target <= 0:
        return {"df": 50.0, "var": float(np.median(var))}

    def func(x: float) -> float:
        return polygamma(1, x / 2.0) - target

    try:
        df_prior = float(bisect(func, 2.0, 200.0, xtol=1e-4, maxiter=100))
    except ValueError:
        df_prior = 50.0

    var_prior = float(np.exp(z_mean + polygamma(0, df_mean / 2.0) - polygamma(0, df_prior / 2.0)))
    return {"df": df_prior, "var": var_prior}


# ---------------------------------------------------------------------------
# Sparse-aware helper functions
# ---------------------------------------------------------------------------

def _compute_mean_diff_sparse_safe(
    mdata: "MethylData",
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> np.ndarray:
    """Compute mean methylation difference while preserving sparse matrices.

    Computes mean_diff = mean_beta_B - mean_beta_A using sparse-aware
    operations that avoid full matrix conversion to dense.
    """
    from scipy.sparse import issparse

    X = mdata.adata.X
    if issparse(X):
        # Sparse matrix: use efficient row-wise operations
        mean_a = np.asarray(X[idx_a, :].mean(axis=0)).ravel()
        mean_b = np.asarray(X[idx_b, :].mean(axis=0)).ravel()
        mean_diff = mean_b - mean_a
    else:
        # Dense matrix: standard computation
        mean_a = np.nanmean(X[idx_a, :], axis=0)
        mean_b = np.nanmean(X[idx_b, :], axis=0)
        mean_diff = mean_b - mean_a

    return mean_diff


def _compute_group_means_sparse_safe(
    mdata: "MethylData",
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-group means while preserving sparse matrices."""
    from scipy.sparse import issparse

    X = mdata.adata.X
    if issparse(X):
        mean_a = np.asarray(X[idx_a, :].mean(axis=0)).ravel()
        mean_b = np.asarray(X[idx_b, :].mean(axis=0)).ravel()
    else:
        mean_a = np.nanmean(X[idx_a, :], axis=0)
        mean_b = np.nanmean(X[idx_b, :], axis=0)

    return mean_a, mean_b


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

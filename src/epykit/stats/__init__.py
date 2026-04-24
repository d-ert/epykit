"""
epykit.stats — Statistical testing engine
=============================================
Implements differential methylation testing at single-base (DMC) and
regional (DMR) resolution.

Statistical models (in order of complexity)
--------------------------------------------
1. **Fisher's Exact Test** — no replicates; exact p-value for 2×2
   contingency table [methylated, unmethylated] × [group_A, group_B].

2. **Logistic GLM + Likelihood Ratio Test** — with replicates;
   full vs reduced Binomial GLM; deviance difference ~ χ².

3. **HC0 overdispersion correction** — fits robust heteroskedasticity-
   consistent covariance (cov_type='HC0') to emulate methylKit's
   quasi-binomial McCullagh-Nelder correction.

4. **BH-FDR + DMR merging** — Benjamini-Hochberg multiple testing
   correction followed by distance-based merging of adjacent DMCs
   into Differentially Methylated Regions.

Master entry point
------------------
>>> from epykit.stats import calculate_diff_meth
>>> results = calculate_diff_meth(mdata, treatment_col="group")
"""

from epykit.stats.dmr import merge_dmrs
from epykit.stats.tests import calculate_diff_meth, fisher_exact_test, glm_lrt_test

__all__ = [
    "calculate_diff_meth",
    "fisher_exact_test",
    "glm_lrt_test",
    "merge_dmrs",
]

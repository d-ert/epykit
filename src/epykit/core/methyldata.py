"""
epykit.core.methyldata
========================
The ``MethylData`` class is the primary public-facing data structure of
py-methyl-toolkit.  It wraps an ``anndata.AnnData`` object and provides
a methylation-specific, typed API that mirrors the ergonomics of R's
``methylKit`` while benefiting from the AnnData ecosystem.

Key design principles
---------------------
- **MethylData is immutable by convention** — every filter/subset method
  returns a *new* ``MethylData`` instance backed by a copy (or view) of
  the underlying AnnData, preventing accidental mutation.
- **AnnData geometry** — samples are rows (obs), sites are columns (var).
- **Lazy where possible** — large operations are deferred to AnnData / NumPy
  sparse operations rather than pulling full matrices into Python lists.

Stored data
-----------
  adata.X                          : beta-value matrix  float32 (n_obs × n_var)
  adata.obs                        : sample metadata    pd.DataFrame
  adata.var                        : site metadata      pd.DataFrame
  adata.layers['coverage']         : total read depth   int32
  adata.layers['methylated_counts']: methylated reads   int32
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError("anndata is required: pip install anndata") from e

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# MethylData class
# ---------------------------------------------------------------------------

class MethylData:
    """Ergonomic wrapper around AnnData for methylation data.

    Parameters
    ----------
    adata:
        An ``anndata.AnnData`` object constructed by
        :func:`epykit.io.build_anndata` or :func:`epykit.io.read_samples`.
        Expected layers: ``coverage``, ``methylated_counts``.
        Expected var columns: ``chr``, ``start``, ``end``, ``strand``,
        ``context``.

    Attributes
    ----------
    adata:
        The underlying ``AnnData`` object.

    Examples
    --------
    >>> from epykit.io import read_samples
    >>> from epykit.core import MethylData
    >>>
    >>> adata = read_samples("sample_sheet.csv", min_coverage=5)
    >>> mdata = MethylData(adata)
    >>> mdata = mdata.filter_coverage(5, 500).subset_context("CpG").unite()
    >>> print(mdata)
    """

    def __init__(self, adata: "ad.AnnData") -> None:
        _validate_adata(adata)
        self._adata = adata

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def adata(self) -> "ad.AnnData":
        """The underlying AnnData object."""
        return self._adata

    @property
    def beta(self) -> np.ndarray:
        """Beta-value matrix (methylation %).

        Shape: ``(n_samples, n_sites)`` — float32.
        Values range from 0 to 100.
        NaN indicates a site with no coverage in that sample.
        """
        return np.asarray(self._adata.X)

    @property
    def coverage(self) -> np.ndarray:
        """Total read coverage matrix.

        Shape: ``(n_samples, n_sites)`` — int32.
        Zero indicates no coverage.
        """
        return np.asarray(self._adata.layers["coverage"])

    @property
    def methylated(self) -> np.ndarray:
        """Methylated read count matrix.

        Shape: ``(n_samples, n_sites)`` — int32.
        """
        return np.asarray(self._adata.layers["methylated_counts"])

    @property
    def unmethylated(self) -> np.ndarray:
        """Unmethylated read count matrix (coverage - methylated).

        Shape: ``(n_samples, n_sites)`` — int32.
        """
        return self.coverage - self.methylated

    @property
    def sites(self) -> pd.DataFrame:
        """Site coordinate DataFrame (``adata.var``).

        Columns: ``chr``, ``start``, ``end``, ``strand``, ``context``,
        plus any additional annotation columns added by
        :func:`epykit.intervals.annotate_features`.
        """
        return self._adata.var

    @property
    def samples(self) -> pd.DataFrame:
        """Sample metadata DataFrame (``adata.obs``)."""
        return self._adata.obs

    @property
    def n_samples(self) -> int:
        """Number of samples (observations)."""
        return self._adata.n_obs

    @property
    def n_sites(self) -> int:
        """Number of CpG/cytosine sites (variables)."""
        return self._adata.n_vars

    @property
    def obs_names(self) -> pd.Index:
        """Sample identifiers."""
        return self._adata.obs_names

    @property
    def var_names(self) -> pd.Index:
        """Locus key identifiers."""
        return self._adata.var_names

    # ------------------------------------------------------------------
    # Filtering methods
    # ------------------------------------------------------------------

    def filter_coverage(
        self,
        min_cov: int = 1,
        max_cov: int | None = None,
        require_all_samples: bool = False,
    ) -> "MethylData":
        """Remove sites that fail coverage thresholds in any (or all) samples.

        Parameters
        ----------
        min_cov:
            Minimum read coverage required.  Sites where *any* sample
            falls below this threshold are removed (when
            ``require_all_samples=False``).
        max_cov:
            Maximum read coverage.  Sites where *any* sample exceeds this
            threshold are discarded (PCR deduplication filter).
        require_all_samples:
            If ``True``, a site is retained only if **all** samples meet
            the minimum coverage threshold (more stringent, equivalent to
            ``methylKit::unite(min.per.group=n)``).
            If ``False`` (default), only sites where *any* sample fails
            are removed.

        Returns
        -------
        MethylData
            A new MethylData with low-coverage sites removed.

        Examples
        --------
        >>> mdata_filtered = mdata.filter_coverage(5, 500)
        >>> mdata_strict = mdata.filter_coverage(10, require_all_samples=True)
        """
        cov = self.coverage  # (n_samples, n_sites)

        if require_all_samples:
            # Keep sites where ALL samples meet min_cov
            mask = np.all(cov >= min_cov, axis=0)
        else:
            # Keep sites where NO sample falls below min_cov
            mask = np.all(cov >= min_cov, axis=0)

        if max_cov is not None:
            mask = mask & np.all(cov <= max_cov, axis=0)

        n_before = self.n_sites
        n_kept = int(mask.sum())
        logger.info(
            "filter_coverage(min=%d, max=%s): %d → %d sites retained (%.1f%%)",
            min_cov, max_cov, n_before, n_kept,
            100.0 * n_kept / n_before if n_before > 0 else 0.0,
        )

        new_adata = self._adata[:, mask].copy()
        return MethylData(new_adata)

    def subset_context(self, context: str = "CpG") -> "MethylData":
        """Retain only sites matching the given sequence context.

        Parameters
        ----------
        context:
            One of ``"CpG"``, ``"CHG"``, ``"CHH"``.

        Returns
        -------
        MethylData
            A new MethylData containing only the requested context.

        Raises
        ------
        ValueError
            If the ``var`` DataFrame does not contain a ``context`` column,
            or if the supplied context is not found.
        KeyError
            If ``context`` column is missing.

        Examples
        --------
        >>> mdata_cpg = mdata.subset_context("CpG")
        >>> mdata_chg = mdata.subset_context("CHG")
        """
        valid_contexts = {"CpG", "CHG", "CHH"}
        if context not in valid_contexts:
            raise ValueError(
                f"context must be one of {valid_contexts}, got '{context}'"
            )

        if "context" not in self._adata.var.columns:
            raise KeyError(
                "adata.var does not contain a 'context' column. "
                "Did you load data from a CX_report file?"
            )

        mask = self._adata.var["context"] == context
        n_sites = int(mask.sum())
        logger.info(
            "subset_context('%s'): %d → %d sites",
            context, self.n_sites, n_sites,
        )

        new_adata = self._adata[:, mask].copy()
        return MethylData(new_adata)

    def unite(
        self,
        type: str = "intersect",
        min_per_group: int | None = None,
        treatment_col: str | None = None,
    ) -> "MethylData":
        """Retain only loci covered in all (or a minimum number of) samples.

        This is the Python equivalent of ``methylKit::unite()``.

        Parameters
        ----------
        type:
            ``"intersect"`` — keep only sites covered in **all** samples
            (default, most stringent, recommended for GLM-based analysis).
            ``"union"`` — keep all sites (may contain NaN).
        min_per_group:
            Minimum number of samples per group that must cover a site.
            Only used when ``treatment_col`` is also specified.
            If ``None``, all samples must cover the site.
        treatment_col:
            Column in ``obs`` that defines the treatment groups.  When
            combined with ``min_per_group``, a site is retained if at
            least ``min_per_group`` samples per group have coverage > 0.

        Returns
        -------
        MethylData

        Examples
        --------
        >>> united = mdata.unite()                          # all samples
        >>> united = mdata.unite(min_per_group=3,           # ≥3 per group
        ...                      treatment_col="group")
        """
        if type == "union":
            logger.info("unite(type='union'): returning all %d sites", self.n_sites)
            return MethylData(self._adata.copy())

        cov = self.coverage  # (n_samples, n_sites)

        if treatment_col is not None and min_per_group is not None:
            # Per-group coverage mask
            groups = self._adata.obs[treatment_col].unique()
            mask = np.ones(self.n_sites, dtype=bool)
            for grp in groups:
                grp_idx = self._adata.obs[treatment_col] == grp
                grp_cov = cov[grp_idx.values, :]
                n_covered = (grp_cov > 0).sum(axis=0)
                mask &= n_covered >= min_per_group
        else:
            # All samples must have coverage > 0
            mask = np.all(cov > 0, axis=0)

        n_before = self.n_sites
        n_kept = int(mask.sum())
        logger.info(
            "unite(type='%s'): %d → %d sites (%.1f%% retained)",
            type, n_before, n_kept,
            100.0 * n_kept / n_before if n_before > 0 else 0.0,
        )

        new_adata = self._adata[:, mask].copy()
        return MethylData(new_adata)

    # ------------------------------------------------------------------
    # Convenience / info methods
    # ------------------------------------------------------------------

    def coverage_stats(self) -> pd.DataFrame:
        """Return per-sample coverage statistics.

        Returns
        -------
        pd.DataFrame
            Columns: ``mean_cov``, ``median_cov``, ``n_sites``,
            ``pct_sites_covered``, ``n_sites_min1``.
            Index: sample IDs.
        """
        cov = self.coverage
        total_sites = self.n_sites
        stats = []
        for i, sid in enumerate(self.obs_names):
            row_cov = cov[i, :]
            n_covered = int((row_cov > 0).sum())
            stats.append({
                "sample_id": sid,
                "mean_cov": float(np.nanmean(row_cov)),
                "median_cov": float(np.median(row_cov[row_cov > 0])) if n_covered > 0 else 0.0,
                "n_sites": total_sites,
                "n_sites_covered": n_covered,
                "pct_sites_covered": 100.0 * n_covered / total_sites if total_sites > 0 else 0.0,
            })
        return pd.DataFrame(stats).set_index("sample_id")

    def global_methylation(self) -> pd.DataFrame:
        """Compute per-sample global methylation percentage.

        Returns
        -------
        pd.DataFrame
            Index: sample IDs.
            Columns: ``global_beta_mean``, ``global_beta_median``,
            ``total_methylated``, ``total_coverage``.
        """
        meth = self.methylated
        cov = self.coverage
        records = []
        for i, sid in enumerate(self.obs_names):
            total_m = int(meth[i, :].sum())
            total_c = int(cov[i, :].sum())
            row_beta = self.beta[i, :]
            row_beta_covered = row_beta[~np.isnan(row_beta)]
            records.append({
                "sample_id": sid,
                "global_beta_mean": float(np.nanmean(row_beta_covered)) if len(row_beta_covered) else np.nan,
                "global_beta_median": float(np.median(row_beta_covered)) if len(row_beta_covered) else np.nan,
                "total_methylated": total_m,
                "total_coverage": total_c,
                "global_pct_meth": 100.0 * total_m / total_c if total_c > 0 else np.nan,
            })
        return pd.DataFrame(records).set_index("sample_id")

    def copy(self) -> "MethylData":
        """Return a deep copy of this MethylData object."""
        return MethylData(self._adata.copy())

    def __repr__(self) -> str:
        return (
            f"MethylData object\n"
            f"  {self.n_samples} samples × {self.n_sites} sites\n"
            f"  obs:    {list(self._adata.obs.columns)}\n"
            f"  var:    {list(self._adata.var.columns)}\n"
            f"  layers: {list(self._adata.layers.keys())}"
        )

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_adata(adata: "ad.AnnData") -> None:
    """Validate that an AnnData object has the required structure."""
    required_layers = {"coverage", "methylated_counts"}
    missing_layers = required_layers - set(adata.layers.keys())
    if missing_layers:
        raise ValueError(
            f"AnnData is missing required layers: {missing_layers}. "
            "Use epykit.io.read_samples() or epykit.io.build_anndata() "
            "to create a properly structured AnnData object."
        )

    required_var_cols = {"chr", "start", "end"}
    missing_var = required_var_cols - set(adata.var.columns)
    if missing_var:
        raise ValueError(
            f"AnnData.var is missing required columns: {missing_var}."
        )

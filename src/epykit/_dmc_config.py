"""The keyword knobs of ``tl.dmc`` carried as one frozen record.

``DMCConfig`` holds every keyword argument ``ep.tl.dmc`` accepts, with the
same names and defaults. The orchestrator builds one per call and uses it
to validate the request, fingerprint the ``resumable=True`` lookup and write
the ``md.uns["dmc"]`` metadata record, so those pieces of logic have a
single owner. The public ``ep.tl.dmc(...)`` signature is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ._dmc_engines import engine_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DMCConfig:
    """Every keyword knob of ``tl.dmc``; see that docstring for meanings.

    ``min_samples_treatment=None`` reads as 0 on construction, as ``tl.dmc``
    has always done.
    """

    test: str = "auto"
    chromosomes: list[str] | None = None
    min_samples_treatment: int | None = None
    min_samples_control: int = 0
    dispersion: str = "eb"
    reference: str = "adaptive"
    allow_n1: bool = False
    formula: str | None = None
    contrast: Any = None
    covariates: list[str] | None = None
    treatment_col: str = "treatment"
    empirical_fdr: bool = False
    n_perm: int = 100
    perm_seed: int = 42
    perm_n_jobs: int = 1
    backend: str = "sequential"
    n_workers: int | None = None
    glm_backend: str = "cpu"
    resumable: bool = False
    materialize: bool = True
    use_smoothed: bool = False
    smoothing: bool = False
    smoothing_span_bp: int = 500
    canonical_only: bool = False
    fdr_method: str = "fdr_bh"
    neighbour_combine: bool = False
    neighbour_bp: int = 500
    sep_fallback: bool = False
    sep_threshold: float = 0.9
    reference_level: str | None = None
    tsv: str | bool | None = None
    tsv_full: bool = False
    tsv_alpha: float = 0.05
    csv: str | None = None
    csv_full: bool = False
    csv_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.min_samples_treatment is None:
            object.__setattr__(self, "min_samples_treatment", 0)

    def validate(self) -> None:
        """Reject removed and unknown engine names.

        Runs before the formula / contrast dispatch, so a bad ``test`` is
        refused on every path before any store is opened. An engine removed
        in 0.7.5 raises with its migration hint, any other unknown name with
        the public choice list. ``"auto"`` passes: the binary path resolves
        it after the n=1 guard.
        """
        if self.test != "auto":
            engine_spec(self.test)

    def validate_resolved(self) -> None:
        """Checks on the resolved configuration before compute starts.

        ``materialize=False`` cannot run the eager-only post-processors
        that need the full in-memory result table.
        """
        if not self.materialize:
            incompatible = [
                name
                for name, on in (
                    ("neighbour_combine", self.neighbour_combine),
                    ("empirical_fdr", self.empirical_fdr),
                    ("use_smoothed", self.use_smoothed),
                )
                if on
            ]
            if incompatible:
                raise ValueError(
                    "materialize=False keeps only the streaming DMCStore handle "
                    "and cannot run features that post-process the full in-memory "
                    f"result table: {', '.join(incompatible)}. Re-run with "
                    "materialize=True (the default) for these, or disable them."
                )

    def resume_signature_params(self, *, selected_test: str, unite: bool) -> dict[str, Any]:
        """The params half of the ``resumable=True`` fingerprint.

        Every knob that changes engine output is listed, including the
        post-processing knobs, ``canonical_only`` and the DSS-style count
        smoothing: leaving one out would let a parameter sweep silently
        reuse a cached result computed at different values. The smoothing
        span is keyed only while smoothing is on, so changing an unused
        span does not invalidate a cache.
        """
        params = {
            "support_schema_revision": "2026-10-09-observed-replicates-v1",
            "test": selected_test,
            "chromosomes": self.chromosomes,
            "canonical_only": self.canonical_only,
            "unite": unite,
            "min_samples_treatment": self.min_samples_treatment,
            "min_samples_control": self.min_samples_control,
            "dispersion": self.dispersion,
            "reference": self.reference,
            "empirical_fdr": self.empirical_fdr,
            "n_perm": self.n_perm if self.empirical_fdr else None,
            "perm_seed": self.perm_seed if self.empirical_fdr else None,
            "sep_fallback": self.sep_fallback,
            "sep_threshold": self.sep_threshold,
            "neighbour_combine": self.neighbour_combine,
            "neighbour_bp": self.neighbour_bp,
            "fdr_method": self.fdr_method,
            "smoothing": self.smoothing,
            "smoothing_span_bp": self.smoothing_span_bp if self.smoothing else None,
        }
        if selected_test == "glm":
            from ._glm import GLM_ALGORITHM_REVISION

            params["glm_algorithm_revision"] = GLM_ALGORITHM_REVISION
        return params

    def to_uns(
        self,
        *,
        test_used: str,
        n_sites: int,
        materialized: bool,
        unite: bool,
        last_key: str,
        store_path: str | None,
        resumed: bool = False,
        smooth_method: str | None = None,
        formula: str | None = None,
        contrast: str | None = None,
        design_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """The ``md.uns["dmc"]`` record, with the same keys on every path.

        Readers: ``MethylData.get_dmc`` / ``.dmc_store`` / ``.save``
        (``last_key``, ``store_path``), ``report.py`` (``test_used``,
        ``test_requested``, ``fdr_method``), ``multiqc_export`` and
        ``pl.dashboard`` (``test_used``, ``n_sites``, ``unite``),
        ``export.export_tables`` and the CLI (``last_key``).

        The formula / contrast path passes ``design_terms`` together with
        the resolved ``formula`` and ``contrast`` label. That path does not
        consume the permutation FDR or smoothing knobs, so they are recorded
        as ``None`` there; the binary path records the contrast fields as
        ``None``. ``canonical_only`` shapes the chromosome universe on every
        path and is recorded as a bool on all three. ``store_path`` is ``None``
        when no DMCStore was opened (the resume cache hit).
        """
        binary = design_terms is None

        def binary_only(value: Any) -> Any:
            return value if binary else None

        metadata = {
            "test_requested": self.test,
            "test_used": test_used,
            "n_sites": n_sites,
            "materialized": materialized,
            "unite": unite,
            "min_samples_treatment": self.min_samples_treatment,
            "min_samples_control": self.min_samples_control,
            "dispersion": self.dispersion,
            "reference": self.reference,
            "empirical_fdr": binary_only(self.empirical_fdr),
            "n_perm": binary_only(self.n_perm if self.empirical_fdr else None),
            "perm_seed": binary_only(self.perm_seed if self.empirical_fdr else None),
            "sep_fallback": binary_only(self.sep_fallback),
            "sep_threshold": binary_only(self.sep_threshold),
            "neighbour_combine": binary_only(self.neighbour_combine),
            "neighbour_bp": binary_only(self.neighbour_bp),
            "fdr_method": self.fdr_method,
            # Explicit pointer so MethylData.get_dmc() / .dmc resolve to the
            # table just written, whatever other tests ran in the session.
            "last_key": last_key,
            "use_smoothed": binary_only(self.use_smoothed),
            "smooth_method": binary_only(smooth_method),
            # DSS-style count smoothing; the span only means something when
            # smoothing is on.
            "smoothing": binary_only(bool(self.smoothing)),
            "smoothing_span_bp": binary_only(
                int(self.smoothing_span_bp) if self.smoothing else None
            ),
            # Opt-in canonical chromosome filter of the auto-detected
            # partitions; applies to the binary and the contrast run alike.
            "canonical_only": bool(self.canonical_only),
            # Persistent per-chromosome DMCStore, so tl.dmr can stream
            # chromosomes from disk instead of holding the full table.
            "store_path": store_path,
            "resumed": resumed,
            "formula": formula,
            "contrast": contrast,
            "design_terms": design_terms,
            "covariates": None if binary else (list(self.covariates) if self.covariates else None),
            "treatment_col": None if binary else self.treatment_col,
        }
        if test_used in {"glm", "glm_contrast"}:
            from ._glm import GLM_ALGORITHM_REVISION

            metadata["glm_algorithm_revision"] = GLM_ALGORITHM_REVISION
        return metadata

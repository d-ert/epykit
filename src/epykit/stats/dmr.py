"""
epykit.stats.dmr
==================
Distance-based merging of adjacent Differentially Methylated Cytosines
(DMCs) into Differentially Methylated Regions (DMRs).

Algorithm
---------
1. Filter the DMC result table by significance threshold (``qvalue < 0.05``).
2. Sort significant sites by chromosome and position.
3. Group consecutive sites where the distance between adjacent sites is ≤
   ``max_gap`` bp and the methylation difference has the same sign.
4. Optionally require a minimum number of sites and a minimum absolute
   methylation difference per DMR.
5. Return a summary DataFrame of merged DMRs.

The output format is compatible with BED format for downstream analysis
in genome browsers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DMR merging
# ---------------------------------------------------------------------------

def merge_dmrs(
    dmc_results: pd.DataFrame,
    *,
    qvalue_cutoff: float = 0.05,
    mean_diff_cutoff: float = 0.0,
    max_gap: int = 1000,
    min_sites: int = 3,
    min_abs_diff: float = 10.0,
    same_direction: bool = True,
) -> pd.DataFrame:
    """Merge adjacent DMCs into DMRs using a distance-based approach.

    This is equivalent to ``methylKit``-style DMR merging.  Consecutive
    significant CpG sites that are within ``max_gap`` bp of each other
    (and optionally have the same direction of methylation change) are
    merged into a single DMR.

    Parameters
    ----------
    dmc_results:
        Output from :func:`epykit.stats.calculate_diff_meth`.
        Required columns: ``chr``, ``start``, ``end``, ``qvalue``,
        ``mean_diff``.
    qvalue_cutoff:
        Significance threshold for including a DMC.  Default: 0.05.
    mean_diff_cutoff:
        Minimum absolute mean methylation difference (%) for a DMC to
        be included.  Default: 0.0 (no filter).
    max_gap:
        Maximum distance in bp between adjacent significant sites to
        consider them part of the same DMR.  Default: 1000 bp.
    min_sites:
        Minimum number of CpG sites required to report a DMR.
        Default: 3.
    min_abs_diff:
        Minimum absolute mean methylation difference (%) for the
        merged DMR to be reported.  Default: 10.0%.
    same_direction:
        If ``True`` (default), only merge sites with the same sign of
        ``mean_diff`` (all hyper- or all hypo-methylated).

    Returns
    -------
    pd.DataFrame
        One row per DMR.  Columns:
        ``chr``, ``start``, ``end``, ``n_cpgs``,
        ``mean_diff``, ``min_qvalue``, ``direction``.

    Examples
    --------
    >>> from epykit.stats import calculate_diff_meth, merge_dmrs
    >>> dmc = calculate_diff_meth(mdata, treatment_col="group")
    >>> dmrs = merge_dmrs(dmc, qvalue_cutoff=0.05, min_sites=3)
    >>> print(f"Found {len(dmrs)} DMRs")
    """
    required = {"chr", "start", "end", "qvalue", "mean_diff"}
    missing = required - set(dmc_results.columns)
    if missing:
        raise ValueError(f"dmc_results is missing columns: {missing}")

    # --- Filter significant DMCs ---
    sig = dmc_results[dmc_results["qvalue"] < qvalue_cutoff].copy()

    if mean_diff_cutoff > 0:
        sig = sig[sig["mean_diff"].abs() >= mean_diff_cutoff]

    if len(sig) == 0:
        logger.info("merge_dmrs: no significant DMCs found (n=0).")
        return pd.DataFrame(
            columns=["chr", "start", "end", "n_cpgs", "mean_diff", "min_qvalue", "direction"]
        )

    logger.info(
        "merge_dmrs: %d significant DMCs → merging (max_gap=%d, min_sites=%d)",
        len(sig), max_gap, min_sites,
    )

    # Sort by chr, start
    sig = sig.sort_values(["chr", "start"]).reset_index(drop=True)

    # --- Distance-based merging ---
    dmr_records = []
    current_chr = None
    current_sites = []

    def flush_region(sites: list[dict]) -> None:
        """Process a collected region and add to dmr_records if valid."""
        if len(sites) < min_sites:
            return
        mean_d = float(np.mean([s["mean_diff"] for s in sites]))
        if abs(mean_d) < min_abs_diff:
            return
        min_q = float(min(s["qvalue"] for s in sites))
        direction = "hyper" if mean_d > 0 else "hypo"
        dmr_records.append({
            "chr": sites[0]["chr"],
            "start": sites[0]["start"],
            "end": sites[-1]["end"],
            "n_cpgs": len(sites),
            "mean_diff": mean_d,
            "min_qvalue": min_q,
            "direction": direction,
        })

    for _, row in sig.iterrows():
        site = {
            "chr": row["chr"],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "mean_diff": float(row["mean_diff"]),
            "qvalue": float(row["qvalue"]),
        }

        if current_chr != site["chr"]:
            # New chromosome — flush current region
            flush_region(current_sites)
            current_chr = site["chr"]
            current_sites = [site]
            continue

        if not current_sites:
            current_sites = [site]
            continue

        prev = current_sites[-1]
        gap = site["start"] - prev["end"]

        # Check direction consistency if required
        same_dir = (
            not same_direction
            or (np.sign(site["mean_diff"]) == np.sign(prev["mean_diff"]))
        )

        if gap <= max_gap and same_dir:
            current_sites.append(site)
        else:
            flush_region(current_sites)
            current_sites = [site]

    # Flush the last region
    flush_region(current_sites)

    if not dmr_records:
        logger.info("merge_dmrs: no DMRs passed filters (min_sites=%d, min_abs_diff=%.1f).",
                    min_sites, min_abs_diff)
        return pd.DataFrame(
            columns=["chr", "start", "end", "n_cpgs", "mean_diff", "min_qvalue", "direction"]
        )

    dmrs = pd.DataFrame(dmr_records).sort_values(["chr", "start"]).reset_index(drop=True)

    n_hyper = int((dmrs["direction"] == "hyper").sum())
    n_hypo = int((dmrs["direction"] == "hypo").sum())
    logger.info(
        "merge_dmrs: %d DMRs found (%d hyper, %d hypo)",
        len(dmrs), n_hyper, n_hypo,
    )

    return dmrs


# ---------------------------------------------------------------------------
# DMR export helpers
# ---------------------------------------------------------------------------

def dmrs_to_bed(
    dmrs: pd.DataFrame,
    path: str,
    *,
    score_col: str = "mean_diff",
    name_col: str = "direction",
    include_header: bool = False,
) -> None:
    """Write a DMR DataFrame to a BED file.

    Parameters
    ----------
    dmrs:
        Output from :func:`merge_dmrs`.
    path:
        Output file path (e.g. ``"dmrs.bed"``).
    score_col:
        Column to use as the BED score (column 5).
    name_col:
        Column to use as the BED name (column 4).
    include_header:
        Whether to include a header row (non-standard BED).

    Examples
    --------
    >>> dmrs_to_bed(dmrs, "output_dmrs.bed")
    """
    bed = pd.DataFrame({
        "chr": dmrs["chr"],
        "start": dmrs["start"],
        "end": dmrs["end"],
        "name": dmrs[name_col] if name_col in dmrs.columns else "DMR",
        "score": (dmrs[score_col].abs() * 10).clip(0, 1000).astype(int)
        if score_col in dmrs.columns else 0,
        "strand": ".",
    })

    bed.to_csv(
        path,
        sep="\t",
        index=False,
        header=include_header,
    )
    logger.info("Wrote %d DMRs to BED file: %s", len(dmrs), path)


def filter_dmrs(
    dmrs: pd.DataFrame,
    *,
    direction: str | None = None,
    min_n_cpgs: int = 1,
    min_abs_diff: float = 0.0,
    max_qvalue: float = 1.0,
    chromosomes: list[str] | None = None,
) -> pd.DataFrame:
    """Filter a DMR DataFrame by various criteria.

    Parameters
    ----------
    dmrs:
        Output from :func:`merge_dmrs`.
    direction:
        ``"hyper"`` for hypermethylated DMRs, ``"hypo"`` for
        hypomethylated, ``None`` for both.
    min_n_cpgs:
        Minimum number of CpG sites in a DMR.
    min_abs_diff:
        Minimum absolute mean methylation difference.
    max_qvalue:
        Maximum minimum q-value.
    chromosomes:
        List of chromosomes to retain.

    Returns
    -------
    pd.DataFrame
        Filtered DMR table.
    """
    out = dmrs.copy()

    if direction is not None:
        out = out[out["direction"] == direction]

    out = out[out["n_cpgs"] >= min_n_cpgs]
    out = out[out["mean_diff"].abs() >= min_abs_diff]
    out = out[out["min_qvalue"] <= max_qvalue]

    if chromosomes is not None:
        out = out[out["chr"].isin(chromosomes)]

    return out.reset_index(drop=True)

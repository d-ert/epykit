import epykit
import os
import time
import traceback
from datetime import datetime

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def _get_rss_mb() -> float:
    """Return resident set size (RSS) in MB."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    # Fallback to /proc/self/status
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


def _log_step(message: str, start_time: float | None = None) -> None:
    """Print a timestamped debug message with memory usage."""
    elapsed = ""
    if start_time is not None:
        elapsed = f" | elapsed: {time.time() - start_time:.2f}s"
    rss_mb = _get_rss_mb()
    rss_info = f" | RSS: {rss_mb:.1f} MB" if rss_mb >= 0 else ""
    print(f"[{datetime.utcnow().isoformat()}] {message}{elapsed}{rss_info}")

"""
End-to-end scratch workflow including regions_bed and BED-style coordinates.
"""

def main() -> None:
    _log_step("Starting scratch workflow with DIAGNOSTICS")
    _log_step("=" * 80)
    _log_step("DIAGNOSTICS ENABLED: Memory profiling at each DuckDB step")
    _log_step("This will help identify which SQL operation hits the RAM limit")
    _log_step("=" * 80)

    # Build cohort with optional region preselection
    # NOTE: Bismark files have data starting at ~13,283 bp in chr1.
    # Using a region that captures real data for testing.
    regions_bed = "scratch_regions.bed"
    with open(regions_bed, "w", encoding="utf-8") as handle:
        # Region with actual coverage: chr1:13k-50k (contains real methylation sites)
        handle.write("chr1\t13283\t50000\n")
    _log_step(f"Regions BED created: {regions_bed} (chr1:13283-50000)")

    read_start = time.time()
    _log_step("Reading samples via epykit.io.read_samples", read_start)
    _log_step("DuckDB settings: memory_limit=14GB, threads=1, validate_output=False")
    adata = epykit.io.read_samples(
        "samplesheet.csv",
        engine="duckdb",
        min_coverage=10,
        regions_bed=regions_bed,
        output="zarr",
        out_path="cohort.zarr",
        duckdb_memory_limit="14GB",
        duckdb_threads=1,
        validate_output=False,  # Skip expensive Zarr re-read during peak I/O
    )
    _log_step(f"Cohort built with shape: {adata.shape}", read_start)

    reload_start = time.time()
    _log_step("Reloading cohort from Zarr")
    # Reload (Option A: epykit helper)
    adata = epykit.io.load("cohort.zarr")
    _log_step(f"Reloaded cohort with shape: {adata.shape}", reload_start)

    from epykit.core import MethylData

    mdata = MethylData(adata)
    _log_step(f"Constructed MethylData: {mdata}")

    from epykit import plot as epyplot

    # coverage distributions
    plot_start = time.time()
    _log_step("Generating coverage_hist plot", plot_start)
    fig = epyplot.coverage_hist(mdata, max_cov=200)
    fig.savefig("coverage_hist.png", dpi=150, bbox_inches="tight")
    _log_step("Saved coverage_hist.png", plot_start)

    # beta distributions
    plot_start = time.time()
    _log_step("Generating methylation_distribution plot", plot_start)
    fig = epyplot.methylation_distribution(mdata)
    fig.savefig("meth_dist.png", dpi=150, bbox_inches="tight")
    _log_step("Saved meth_dist.png", plot_start)

    # PCA / sample correlation
    plot_start = time.time()
    _log_step("Generating PCA plot", plot_start)
    fig = epyplot.pca(mdata.adata, color_by="group")  # uses adata.obs['group']
    fig.savefig("pca.png", dpi=150, bbox_inches="tight")
    _log_step("Saved pca.png", plot_start)

    plot_start = time.time()
    _log_step("Generating sample correlation plot", plot_start)
    fig = epyplot.sample_correlation(mdata.adata)
    fig.savefig("sample_corr.png", dpi=150, bbox_inches="tight")
    _log_step("Saved sample_corr.png", plot_start)

    filter_start = time.time()
    _log_step("Filtering MethylData")
    mdata = (
        mdata
        .filter_coverage(min_cov=10, max_cov=20)   # tune to your dataset
        .subset_context("CpG")                      # if context exists
        .unite(type="intersect")                    # keep sites covered in all samples
    )
    _log_step("Filtering complete", filter_start)

    # coverage distributions
    plot_start = time.time()
    _log_step("Generating filtered coverage_hist plot", plot_start)
    fig = epyplot.coverage_hist(mdata, max_cov=200)
    fig.savefig("coverage_hist_filt.png", dpi=150, bbox_inches="tight")
    _log_step("Saved coverage_hist_filt.png", plot_start)

    # beta distributions
    plot_start = time.time()
    _log_step("Generating filtered methylation_distribution plot", plot_start)
    fig = epyplot.methylation_distribution(mdata)
    fig.savefig("meth_dist_filt.png", dpi=150, bbox_inches="tight")
    _log_step("Saved meth_dist_filt.png", plot_start)

    _log_step("Workflow completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_step("Workflow failed with exception")
        traceback.print_exc()

"""
epykit.io.anndata_builder_duckdb
=================================
Memory-efficient AnnData construction using DuckDB as the out-of-core
query engine.  No Zarr, no temp files, no intermediate disk writes.

Why DuckDB instead of the old approach
---------------------------------------
The original builders load all N sample DataFrames into Python RAM
simultaneously before computing the locus union.  For 6 Bismark files
with 18–29 M sites each the peak footprint is roughly::

    6 files × 29 M rows × 7 cols × 8 bytes  ≈ 14 GB  (DataFrames)
  + 42 M sites × 3 matrices × 4 bytes        ≈  3 GB  (output)
  ─────────────────────────────────────────────────────────────────
  ≈ 17–18 GB peak  →  crashes on most workstations

How this module works
----------------------
DuckDB is an in-process columnar OLAP engine that reads gzip-compressed
CSV files natively — data never passes through the Python interpreter
until we explicitly fetch a result.

Algorithm::

    Step 1 — DuckDB computes the UNION / INTERSECT of all loci
              across all files.  Entirely inside DuckDB's buffer
              pool; Python sees nothing.

    Step 2 — Pre-allocate three NumPy output arrays of shape
              (n_samples, n_sites).  This is the irreducible minimum —
              the output itself.

    Step 3 — For each sample (one at a time):
                DuckDB LEFT JOINs the locus index against the
                sample's gzip file (streamed from disk on the fly).
                fetchnumpy() pulls the result into Python (~1.3 GB),
                it is scattered into the pre-allocated row, then freed.

Peak RAM breakdown::

    Output arrays          :  n_samples × n_sites × 3 × 4 bytes
                           =  6 × 42 M × 3 × 4 bytes  ≈  3.0 GB
    One sample JOIN result :  n_sites × 4 cols × 8 bytes ≈  1.3 GB
    DuckDB buffer pool     :  configurable, default 2 GB
    ──────────────────────────────────────────────────────────────────
    Total peak             :  ≈ 4.5 – 6.3 GB   (vs. 17–18 GB before)

No Zarr, no temp files, no extra disk I/O.

Requirements
------------
    pip install duckdb      # only new dependency
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from epykit.io.regions import merge_bed_intervals, read_bed_regions

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError("anndata is required: pip install anndata") from e

try:
    import duckdb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "duckdb is required for the streaming builder.\n"
        "Install it with:\n"
        "  pip install duckdb\n"
        "or add the optional extra:\n"
        "  pip install 'epykit[streaming]'"
    ) from e


# ---------------------------------------------------------------------------
# Memory profiling utilities for diagnostics
# ---------------------------------------------------------------------------

import os
import time
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


def _log_duckdb_memory_state(con: "duckdb.DuckDBPyConnection", label: str) -> None:
    """Log current memory state and DuckDB configuration."""
    rss_mb = _get_rss_mb()
    rss_info = f" | RSS: {rss_mb:.1f} MB" if rss_mb >= 0 else ""
    
    # Query DuckDB for actual settings
    try:
        mem_limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        thread_count_result = con.execute(
            "SELECT current_setting('threads')"
        ).fetchone()
        thread_count = thread_count_result[0] if thread_count_result else "unknown"
    except Exception:
        mem_limit = "query failed"
        thread_count = "query failed"
    
    timestamp = datetime.utcnow().isoformat()
    print(
        f"[{timestamp}] {label} | mem_limit={mem_limit} | threads={thread_count}{rss_info}"
    )


# Chromosome sort order for human (and mouse) genomes.
# Unknown contigs are sorted last (key = 99).
_CHR_SORT_EXPR = """
    CASE chr
        WHEN 'chr1'  THEN  1  WHEN 'chr2'  THEN  2
        WHEN 'chr3'  THEN  3  WHEN 'chr4'  THEN  4
        WHEN 'chr5'  THEN  5  WHEN 'chr6'  THEN  6
        WHEN 'chr7'  THEN  7  WHEN 'chr8'  THEN  8
        WHEN 'chr9'  THEN  9  WHEN 'chr10' THEN 10
        WHEN 'chr11' THEN 11  WHEN 'chr12' THEN 12
        WHEN 'chr13' THEN 13  WHEN 'chr14' THEN 14
        WHEN 'chr15' THEN 15  WHEN 'chr16' THEN 16
        WHEN 'chr17' THEN 17  WHEN 'chr18' THEN 18
        WHEN 'chr19' THEN 19  WHEN 'chr20' THEN 20
        WHEN 'chr21' THEN 21  WHEN 'chr22' THEN 22
        WHEN 'chrX'  THEN 23  WHEN 'chrY'  THEN 24
        WHEN 'chrM'  THEN 25  WHEN 'chrMT' THEN 25
        ELSE 99
    END
"""


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_anndata_streaming(
    sample_ids: list[str],
    file_paths: list[PathLike],
    obs_metadata: pd.DataFrame | None = None,
    *,
    min_coverage: int = 1,
    max_coverage: int | None = None,
    join_type: str = "outer",
    duckdb_memory_limit: str = "2GB",
    duckdb_threads: int | None = None,
    fill_beta_na: float = float("nan"),
    fill_counts_na: int = 0,
    regions_bed: PathLike | None = None,
) -> "ad.AnnData":
    """Build a cohort AnnData from Bismark files with minimal peak RAM.

    DuckDB reads gzip files natively and performs the locus union/intersection
    entirely out-of-core.  Python only ever holds one sample's JOIN result
    in RAM at a time, plus the pre-allocated output arrays.

    Parameters
    ----------
    sample_ids:
        Ordered list of sample identifiers (become ``adata.obs_names``).
    file_paths:
        Paths to Bismark coverage files — plain or gzip-compressed.
        Expected format (no header, tab-separated)::

            chr  start  end  methylation_%  count_methylated  count_unmethylated

    obs_metadata:
        Optional pandas DataFrame with sample metadata, indexed by sample_id.
        Stored in ``adata.obs``.
    min_coverage:
        Minimum total read coverage to include a site.  Filtering happens
        inside DuckDB — rows below the threshold never enter Python RAM.
    max_coverage:
        Maximum total read coverage (PCR dedup filter).
    join_type:
        ``"outer"`` — keep all loci found in at least one sample (default).
        ``"inner"`` — keep only loci present in every sample.
    duckdb_memory_limit:
        Maximum RAM DuckDB may use for its internal buffer pool.
        Reduce this if you are very RAM-constrained.  Default ``"2GB"``.
    fill_beta_na:
        Value written for sites absent in a sample.  Default ``NaN``.
    fill_counts_na:
        Value written for absent coverage/count cells.  Default ``0``.
    regions_bed:
        Optional BED file (0-based, half-open) to restrict loci during
        union/intersection and sample joins.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_sites)`` with::

            X                           : beta-value matrix  float32
            layers["coverage"]          : total read depth   int32
            layers["methylated_counts"] : methylated reads   int32

    Examples
    --------
    >>> from epykit.io import build_anndata_streaming
    >>> adata = build_anndata_streaming(
    ...     sample_ids=["ctrl_1", "cd55_1", "ctrl_2", "cd55_2"],
    ...     file_paths=[
    ...         "ctrl_1.bismark.cov.gz",
    ...         "cd55_1.bismark.cov.gz",
    ...         "ctrl_2.bismark.cov.gz",
    ...         "cd55_2.bismark.cov.gz",
    ...     ],
    ...     min_coverage=10,
    ... )
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if len(sample_ids) != len(file_paths):
        raise ValueError(
            f"sample_ids ({len(sample_ids)}) and file_paths ({len(file_paths)}) "
            "must have the same length."
        )
    if not sample_ids:
        raise ValueError("At least one sample is required.")
    if join_type not in ("outer", "inner"):
        raise ValueError(f"join_type must be 'outer' or 'inner', got {join_type!r}")

    file_paths = [Path(p) for p in file_paths]
    for p in file_paths:
        if not p.exists():
            raise FileNotFoundError(f"Sample file not found: {p}")

    n_samples = len(sample_ids)
    logger.info(
        "build_anndata_streaming: %d samples | join='%s' | duckdb_mem=%s",
        n_samples, join_type, duckdb_memory_limit,
    )

    # ------------------------------------------------------------------
    # Open DuckDB connection
    # ------------------------------------------------------------------
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{duckdb_memory_limit}';")
    
    # Set threads: use provided param or default to CPU count
    if duckdb_threads is None:
        import os
        duckdb_threads = os.cpu_count() or 4
    con.execute(f"SET threads TO {duckdb_threads};")
    
    # Log initial state
    _log_duckdb_memory_state(con, "[INIT] DuckDB connection opened")

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------

    regions_filter_sql = ""
    if regions_bed is not None:
        regions_df = merge_bed_intervals(read_bed_regions(regions_bed))
        if not regions_df.empty:
            # NOTE: DuckDB treats END as a reserved keyword, so we quote it.
            con.execute("CREATE TEMP TABLE _regions (chr VARCHAR, start INTEGER, \"end\" INTEGER);")
            con.register("_regions_df", regions_df)
            con.execute("INSERT INTO _regions SELECT * FROM _regions_df;")
            regions_filter_sql = (
                "AND EXISTS (SELECT 1 FROM _regions r "
                "WHERE r.chr = chr AND r.start <= start AND start < r.\"end\")"
            )

    def _cov_filter() -> str:
        parts = [f"(methylated + unmethylated) >= {min_coverage}"]
        if max_coverage is not None:
            parts.append(f"(methylated + unmethylated) <= {max_coverage}")
        return " AND ".join(parts)

    def _read_sql(path: Path, extra_cols: str = "") -> str:
        """SQL fragment: read one Bismark .cov[.gz] file via DuckDB."""
        cols = "chr, start" + (f", {extra_cols}" if extra_cols else "")
        return f"""
            SELECT {cols}
            FROM read_csv(
                '{path.as_posix()}',
                delim    = '\t',
                header   = false,
                columns  = {{
                    'chr':          'VARCHAR',
                    'start':        'BIGINT',
                    '_end':         'BIGINT',
                    'beta':         'DOUBLE',
                    'methylated':   'INTEGER',
                    'unmethylated': 'INTEGER'
                }},
                parallel = true
            )
            WHERE {_cov_filter()}
            {regions_filter_sql}
        """

    # ------------------------------------------------------------------
    # Step 1 — Build global locus index entirely inside DuckDB
    #          STREAMING APPROACH: UNION files one-at-a-time to minimize
    #          peak buffer usage. Avoids materializing all 6 files at once.
    # ------------------------------------------------------------------
    logger.info("  [1/3] Building locus index via DuckDB (streaming) …")

    set_op = "UNION" if join_type == "outer" else "INTERSECT"
    
    # Log before starting locus union
    _log_duckdb_memory_state(con, f"[STEP 1] Before locus UNION ({len(file_paths)} files)")

    # Build locus index incrementally by UNIONing files one at a time.
    # This prevents DuckDB from materializing all files simultaneously in the buffer pool.
    # Peak memory per iteration: ~size of one sample + output, not sum of all samples.
    for idx, path in enumerate(file_paths, 1):
        logger.debug(f"  [1/3.{idx}] Processing {path.name} ({idx}/{len(file_paths)})")
        
        if idx == 1:
            # First file: initialize locus set
            con.execute(f"""
                CREATE TEMP TABLE _loci_temp AS
                SELECT chr, start FROM ({_read_sql(path)})
            """)
        else:
            # Subsequent files: compute union/intersection in temp table, then swap.
            # We create in _loci_new first (avoiding reference to dropped table),
            # then replace _loci_temp. This streams the new file and frees old buffer.
            con.execute(f"""
                CREATE TEMP TABLE _loci_new AS
                SELECT chr, start FROM _loci_temp
                {set_op}
                SELECT chr, start FROM ({_read_sql(path)})
            """)
            con.execute("DROP TABLE _loci_temp")
            con.execute("ALTER TABLE _loci_new RENAME TO _loci_temp")
            # Explicitly trigger garbage collection on Python side
            gc.collect()
        
        _log_duckdb_memory_state(con, f"[STEP 1.{idx}] After file {idx}/{len(file_paths)}")

    # Finalize: sort and add locus indices
    con.execute(f"""
        CREATE TEMP TABLE _loci AS
        SELECT
            chr,
            start,
            CAST(
                ROW_NUMBER() OVER (
                    ORDER BY ({_CHR_SORT_EXPR}), start
                ) - 1
            AS INTEGER) AS locus_idx
        FROM _loci_temp
    """)
    con.execute("DROP TABLE _loci_temp")

    n_sites: int = con.execute("SELECT COUNT(*) FROM _loci").fetchone()[0]
    logger.info("  [1/3] Locus index: %d sites.", n_sites)
    
    # Log after locus union
    _log_duckdb_memory_state(con, f"[STEP 1] After locus UNION completed ({n_sites} sites)")

    # ------------------------------------------------------------------
    # Step 2 — Pre-allocate output arrays
    #           This is the unavoidable minimum — the output data itself.
    # ------------------------------------------------------------------
    logger.info(
        "  [2/3] Pre-allocating output arrays (%d × %d) …", n_samples, n_sites
    )

    beta_mat = np.full((n_samples, n_sites), fill_beta_na, dtype=np.float32)
    cov_mat  = np.zeros((n_samples, n_sites), dtype=np.int32)
    meth_mat = np.zeros((n_samples, n_sites), dtype=np.int32)
    
    _log_duckdb_memory_state(con, f"[STEP 2] Arrays pre-allocated ({n_samples} × {n_sites})")

    # ------------------------------------------------------------------
    # Step 3 — Fill one row per sample
    #           DuckDB streams each gzip file on demand; Python holds at
    #           most one sample's JOIN result in RAM at a time.
    # ------------------------------------------------------------------
    logger.info("  [3/3] Filling matrices — one sample at a time …")
    _log_duckdb_memory_state(con, "[STEP 3] Starting sample JOIN loop")

    for i, (sid, path) in enumerate(zip(sample_ids, file_paths)):
        logger.info("    [%d/%d] %s", i + 1, n_samples, sid)
        _log_duckdb_memory_state(con, f"[STEP 3.{i}] Before JOIN for sample {sid}")

        # DuckDB INNER JOINs the locus table against the sample file on disk.
        # Only loci present in the sample are returned (no NULLs/NaNs).
        # fetchnumpy() peak RAM is proportional to sample coverage, not union size.
        result = con.execute(f"""
            SELECT
                l.locus_idx,
                CAST(s.beta                          AS FLOAT)   AS beta,
                CAST(s.methylated                    AS INTEGER)  AS methylated,
                CAST(s.methylated + s.unmethylated   AS INTEGER)  AS coverage
            FROM _loci l
            INNER JOIN (
                {_read_sql(path, extra_cols="beta, methylated, unmethylated")}
            ) s ON l.chr = s.chr AND l.start = s.start
        """).fetchnumpy()

        # Scatter covered loci into preallocated matrices.
        # locus_idx tells us which rows in the output to fill.
        locus_idx = result["locus_idx"]
        beta_mat[i, locus_idx] = result["beta"].astype(np.float32)
        meth_mat[i, locus_idx] = result["methylated"].astype(np.int32)
        cov_mat[i, locus_idx]  = result["coverage"].astype(np.int32)

        # Free the intermediate result before the next sample.
        del result, locus_idx
        gc.collect()
        
        _log_duckdb_memory_state(con, f"[STEP 3.{i}] After JOIN completed and freed for {sid}")

    # ------------------------------------------------------------------
    # Build var DataFrame from the locus index
    # ------------------------------------------------------------------
    logger.info("  Building var DataFrame …")
    
    # KEY INSIGHT: Don't materialize 42M chromosome *strings* — instead fetch
    # integer *codes* from DuckDB and use pd.Categorical.from_codes().
    # Avoids 2+ GB of string overhead while achieving the same Categorical result.
    
    # Step 4a: Build a chromosome code mapping in DuckDB (only ~25 unique values)
    chr_mapping_result = con.execute("""
        SELECT DISTINCT chr, 
               DENSE_RANK() OVER (
                   ORDER BY CASE chr
                       WHEN 'chr1'  THEN  1  WHEN 'chr2'  THEN  2
                       WHEN 'chr3'  THEN  3  WHEN 'chr4'  THEN  4
                       WHEN 'chr5'  THEN  5  WHEN 'chr6'  THEN  6
                       WHEN 'chr7'  THEN  7  WHEN 'chr8'  THEN  8
                       WHEN 'chr9'  THEN  9  WHEN 'chr10' THEN 10
                       WHEN 'chr11' THEN 11  WHEN 'chr12' THEN 12
                       WHEN 'chr13' THEN 13  WHEN 'chr14' THEN 14
                       WHEN 'chr15' THEN 15  WHEN 'chr16' THEN 16
                       WHEN 'chr17' THEN 17  WHEN 'chr18' THEN 18
                       WHEN 'chr19' THEN 19  WHEN 'chr20' THEN 20
                       WHEN 'chr21' THEN 21  WHEN 'chr22' THEN 22
                       WHEN 'chrX'  THEN 23  WHEN 'chrY'  THEN 24
                       WHEN 'chrM'  THEN 25  WHEN 'chrMT' THEN 25
                       ELSE 99
                   END
               ) - 1 AS chr_code
        FROM _loci
        ORDER BY chr_code
    """).fetchall()
    
    chr_categories = [row[0] for row in chr_mapping_result]
    logger.debug(f"  Chromosome categories: {chr_categories}")
    
    # Step 4b: Fetch chr codes (integers), start, and compute end efficiently
    # Pre-allocate arrays for int32 codes and int64 starts, then convert
    chr_codes_list = []
    start_list = []
    
    chunk_size = 2_000_000  # 2M rows per chunk (back to larger, safe with int fetches)
    offset = 0
    
    while offset < n_sites:
        chunk_limit = min(chunk_size, n_sites - offset)
        loci_chunk = con.execute(f"""
            SELECT
                DENSE_RANK() OVER (
                    ORDER BY CASE chr
                        WHEN 'chr1'  THEN  1  WHEN 'chr2'  THEN  2
                        WHEN 'chr3'  THEN  3  WHEN 'chr4'  THEN  4
                        WHEN 'chr5'  THEN  5  WHEN 'chr6'  THEN  6
                        WHEN 'chr7'  THEN  7  WHEN 'chr8'  THEN  8
                        WHEN 'chr9'  THEN  9  WHEN 'chr10' THEN 10
                        WHEN 'chr11' THEN 11  WHEN 'chr12' THEN 12
                        WHEN 'chr13' THEN 13  WHEN 'chr14' THEN 14
                        WHEN 'chr15' THEN 15  WHEN 'chr16' THEN 16
                        WHEN 'chr17' THEN 17  WHEN 'chr18' THEN 18
                        WHEN 'chr19' THEN 19  WHEN 'chr20' THEN 20
                        WHEN 'chr21' THEN 21  WHEN 'chr22' THEN 22
                        WHEN 'chrX'  THEN 23  WHEN 'chrY'  THEN 24
                        WHEN 'chrM'  THEN 25  WHEN 'chrMT' THEN 25
                        ELSE 99
                    END
                ) - 1 AS chr_code,
                start
            FROM _loci
            ORDER BY locus_idx
            LIMIT {chunk_limit} OFFSET {offset}
        """).fetchnumpy()
        
        # Accumulate integer codes and starts (much smaller than strings)
        chr_codes_list.append(loci_chunk["chr_code"].astype(np.int8))
        start_list.append(loci_chunk["start"].astype(np.int32))
        
        del loci_chunk
        gc.collect()
        
        offset += chunk_limit
        logger.debug(f"  Fetched {offset}/{n_sites} loci")
        _log_duckdb_memory_state(con, f"[STEP 4.chunk] Fetched {offset}/{n_sites} loci chunk")
    
    _log_duckdb_memory_state(con, "[STEP 4] Before DuckDB connection closed")
    con.close()
    del con
    gc.collect()
    
    # Step 4c: Concatenate arrays (safe with integers; small memory footprint)
    chr_codes_arr = np.concatenate(chr_codes_list, dtype=np.int8)
    start_arr = np.concatenate(start_list, dtype=np.int32)
    del chr_codes_list, start_list
    gc.collect()
    
    end_vals = (start_arr.astype(np.int64) + 1).astype(np.int32)
    gc.collect()
    
    # Step 4d: Use from_codes to avoid materializing string values
    var_df = pd.DataFrame(index=pd.RangeIndex(n_sites, name="locus_idx"))
    var_df["chr"]      = pd.Categorical.from_codes(chr_codes_arr, categories=chr_categories)
    var_df["start"]    = start_arr
    var_df["end"]      = end_vals
    
    # Create categorical columns with pre-computed codes
    var_df["strand"]   = pd.Categorical.from_codes(
        np.zeros(n_sites, dtype=np.int8),
        categories=["*"]
    )
    var_df["context"]  = pd.Categorical.from_codes(
        np.zeros(n_sites, dtype=np.int8),
        categories=["CpG"]
    )
    var_df["locus_id"] = np.arange(n_sites, dtype=np.int64)
    
    del chr_codes_arr, start_arr, end_vals
    gc.collect()
    rss_before_adata = _get_rss_mb()

    # ------------------------------------------------------------------
    # Build obs DataFrame
    # ------------------------------------------------------------------
    obs_df = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    if obs_metadata is not None:
        obs_df = obs_metadata.reindex(sample_ids)
        obs_df.index.name = "sample_id"

    # ------------------------------------------------------------------
    # Assemble and return AnnData
    # ------------------------------------------------------------------
    adata = ad.AnnData(
        X=beta_mat,
        obs=obs_df,
        var=var_df,
        layers={
            "coverage":          cov_mat,
            "methylated_counts": meth_mat,
        },
    )

    rss_after_adata = _get_rss_mb()
    print(f"[AnnData assembly] RSS delta: {rss_after_adata - rss_before_adata:.1f} MB")

    logger.info(
        "build_anndata_streaming complete: %d samples × %d sites",
        adata.n_obs, adata.n_vars,
    )
    return adata


# ---------------------------------------------------------------------------
# Convenience wrapper — drop-in replacement for read_samples()
# ---------------------------------------------------------------------------

def read_samples_streaming(
    sample_sheet: PathLike,
    *,
    min_coverage: int = 1,
    max_coverage: int | None = None,
    join_type: str = "outer",
    duckdb_memory_limit: str = "2GB",
) -> "ad.AnnData":
    """Load a cohort via a sample sheet CSV with minimal RAM usage.

    Drop-in replacement for :func:`epykit.io.read_samples` for large
    datasets.  Accepts the same sample sheet CSV format::

        sample_id, path, group [, batch, age, ...]

    All columns beyond ``path`` become ``adata.obs`` metadata.

    Parameters
    ----------
    sample_sheet:
        Path to the CSV sample sheet.
    min_coverage, max_coverage, join_type, duckdb_memory_limit:
        Forwarded to :func:`build_anndata_streaming`.

    Returns
    -------
    anndata.AnnData

    Examples
    --------
    >>> from epykit.io import read_samples_streaming
    >>> adata = read_samples_streaming(
    ...     "cohort.csv",
    ...     min_coverage=10,
    ... )
    """
    sample_sheet = Path(sample_sheet)
    if not sample_sheet.exists():
        raise FileNotFoundError(f"Sample sheet not found: {sample_sheet}")

    ss = pd.read_csv(sample_sheet)
    missing = {"sample_id", "path"} - set(ss.columns)
    if missing:
        raise ValueError(
            f"Sample sheet missing required columns: {missing}. "
            f"Found: {list(ss.columns)}"
        )

    for _, row in ss.iterrows():
        p = Path(row["path"])
        if not p.exists():
            raise FileNotFoundError(
                f"Sample file not found for '{row['sample_id']}': {p}"
            )

    sample_ids = list(ss["sample_id"])
    file_paths = [Path(row["path"]) for _, row in ss.iterrows()]
    obs_df = ss[[c for c in ss.columns if c != "path"]].set_index("sample_id")

    return build_anndata_streaming(
        sample_ids=sample_ids,
        file_paths=file_paths,
        obs_metadata=obs_df,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        join_type=join_type,
        duckdb_memory_limit=duckdb_memory_limit,
    )

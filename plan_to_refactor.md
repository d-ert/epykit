## Small Todo
- [x] Add Parquet converter and store backend
- [x] Wire sample-sheet Parquet orchestration
- [x] Validate with uv and smoke tests
- [ ] Move to Phase 2: chromwise DMC processor

## Plan: Migrate EpyKit from AnnData to Parquet-based Architecture

**TL;DR**: This is a **significant refactor but highly manageable** (~70–80% of your existing code can stay). The current codebase is well-designed; you already use Polars extensively and your stats functions are vectorized. The main work is:
1. **New I/O layer** (Parquet conversion + per-chromosome reader)
2. **Refactor DMC pipeline** to process chromosome-by-chromosome instead of loading full adata.X
3. **Adapt existing stats functions** to accept chunked count arrays
4. **Rewrite 5 workflow scripts** to use the new API

You'll keep ~70% of the logic (statistical tests, DMR merging, tiling), replace ~30% (I/O builders, pipeline orchestration), and add ~10% (Parquet-specific layers).

---

### **Steps**

#### **Phase 1: Foundation – Parquet I/O Layer**

1. **Create new I/O module** `src/epykit/io/parquet_converter.py`
   - Function: `convert_sample(input_path, sample_name, output_dir, chunksize=2_000_000)` — mirrors the pseudo-code in your proposal
   - Read Bismark/bedGraph lazily with Polars predicate pushdown
   - Write partitioned Parquet: `output_dir/sample={sample}/chrom={chrom}/part-*.parquet`
   - Handle multiple Bismark input formats (cov.gz, CX_report, bedGraph)
   - See reference: bismark.py for format detection
   
2. **Create data backend layer** `src/epykit/core/parquet_backend.py` (NEW FILE)
   - Class: `ParquetMethylStore(base_dir)` — handles reading, filtering, pivoting
   - Methods:
     - `get_chromosomes()` → list of chromosomes in store
     - `load_chromosome(chrom, samples, filter_specs)` → Polars DataFrame for one chrom (lazy or collected)
     - `filter_coverage(min_cov, max_cov_quantile)` → returns filtered store path
     - `get_sample_summary(sample)` → QC stats (n_CpGs, mean_coverage, global_methylation)
   - This replaces `MethylData` as the primary data accessor
   
3. **Update** sample_sheet.py
   - Add `read_samples_to_parquet()` function — orchestrates batch conversion
   - Takes sample sheet → SBATCH-compatible CLI or direct Python call
   - Parallelizes with `concurrent.futures.ProcessPoolExecutor` (configurable n_workers)
   
4. **Update** methyldata.py
   - Keep `MethylData` class for backwards compatibility (wraps `ParquetMethylStore`)
   - OR replace entirely with Parquet-native API (simpler)
   - Recommendation: **Replace entirely** — MethylData was only needed for AnnData convenience

---

#### **Phase 2: Statistics Engine – Chromosome-wise DMC**

5. **Create per-chromosome DMC processor** `src/epykit/stats/dmr_processor_chromwise.py` (NEW FILE)
   - Function: `process_chromosome(methylstore, chrom, samples, group_a_idxs, group_b_idxs, test, n_threads)`
   - Logic:
     1. Load chrom from all samples (lazy scan, ~50–500 MB per chrom for 50–500 samples)
     2. Filter to sites present in all samples (inner join; or outer join with NaN fill)
     3. Pivot to wide format: one row = one CpG, columns = per-sample N_meth / coverage
     4. Extract count arrays for group A and B
     5. Call existing `tests.py` functions (Fisher/GLM/Limma) unchanged
     6. Add columns: pvalue, effect_size, mean_diff
   - Returns: Polars DataFrame with per-site results
   - **Depends on**: Phase 1 (ParquetMethylStore), existing tests.py
   
6. **Update** tests.py
   - Change signature of `fisher_exact_test()`, `glm_lrt_test()`, `limma_ebayes_test()` to accept:
     - `meth_counts_a`, `cov_counts_a` (numpy arrays, shape (n_sites, n_reps_a))
     - `meth_counts_b`, `cov_counts_b` (arrays, shape (n_sites, n_reps_b))
   - Instead of: full adata.X, adata.layers, design matrices
   - Most of the **internal logic stays the same** (scipy.stats.hypergeom, statsmodels GLM, etc.)
   - Only the **input shape changes** — from (n_samples, n_sites) to iteratively per-chrom
   - Estimate: **50–100 lines changed per function**
   
7. **Create DMC orchestrator** `src/epykit/stats/calculate_diff_meth_chromwise.py` (NEW FILE)
   - Function: `calculate_diff_meth_chromwise(methylstore, samples, group_col, metadata, test="auto", n_threads=None, output_path=None)`
   - Loop over chromosomes in parallel (concurrent.futures or multiprocessing):
     ```python
     with ThreadPoolExecutor(max_workers=n_threads) as pool:
         results = pool.map(
             partial(process_chromosome, methylstore, ..., test=test),
             store.get_chromosomes()
         )
     ```
   - Collect results into one DataFrame per chrom
   - Concatenate and apply **genome-wide FDR** (Benjamini-Hochberg)
   - Optionally write to Parquet or TSV
   - **Replaces**: `stats.calculate_diff_meth()` in the workflow
   
8. **Adapt** glm_vectorized.py
   - No changes needed; it already works on arrays
   - If any densification occurs, refactor to stay vectorized

---

#### **Phase 3: Post-processing – DMR & Tiling**

9. **Update** dmr.py
   - Input: Polars DataFrame with columns [chrom, pos, pvalue, meth_diff, ...]
   - Logic unchanged (`merge_dmrs()` already works on arrays/DataFrames)
   - Output: DMR regions (Parquet or BED)
   
10. **Refactor** tiling.py
    - Currently works with AnnData; adapt to Parquet backend
    - Function: `tile_counts(methylstore, chrom, window_size, step_size)` → returns Polars DataFrame
    - For each window, aggregate counts and coverage across samples
    - Call DMC test on tiled data using `dmr_processor_chromwise.py`
    - Alternative: **Pre-compute tiled windows** during conversion (Phase 1) for speed
    - Recommendation: **Post-hoc tiling during analysis** (more flexible) — see `process_chromosome()` but aggregate by window

---

#### **Phase 4: Visualization**

11. **Create Parquet-native QC module** `src/epykit/plot/qc_parquet.py` (NEW FILE)
    - `sample_summary(methylstore)` → summary stats per sample/chrom
    - `plot_coverage_distribution(methylstore, sample)` → histogram (Polars aggregation → matplotlib)
    - `plot_methylation_distribution(methylstore, sample)` → beta value histogram
    - For **PCA and correlation heatmaps**, optionally load a filtered, small AnnData on-the-fly:
      - Select high-coverage sites (e.g., 50k–100k shared sites)
      - Pivot to X matrix
      - Create minimal AnnData with scanpy for PCA + plotting
      - This keeps visualization code simple and reuses existing plots
    
12. **Add datashader integration** `src/epykit/plot/datashader_plots.py` (NEW FILE)
    - `plot_dmcs_scatter_rasterized(dmcs_df, color_by="qvalue", output_path=None)` → hires scatter plot of DMCs
    - Uses datashader + holoviews for millions of points without RAM blowup
    - Replace qc.py scatter plots

---

#### **Phase 5: Workflow Scripts & Testing**

13. **Rewrite workflow scripts** scripts
    - load_samples.py → call `parquet_converter.convert_sample()` + `sample_summary()`
    - diff_meth.py → call `calculate_diff_meth_chromwise()` → merge_dmrs
    - tile_diff_meth.py → tile within `process_chromosome()` or pre-aggregate
    - qc_plots.py → use `qc_parquet.py` + optional on-the-fly AnnData for PCA
    - merge_dmrs.py → no changes (already works on arrays)
    - Estimate: **Each script shrinks by ~30–50%** (less AnnData boilerplate, cleaner Parquet API)
    
14. **Update tests** tests
    - test_io.py → add tests for `parquet_converter.py`, `ParquetMethylStore`
    - test_stats.py → update to pass chromosome-wise count arrays instead of full adata
    - test_core.py → replace MethylData tests with ParquetMethylStore tests
    - test_intervals.py → update for tiling on Parquet backend
    - Existing test data (small samples) can be converted to Parquet in `conftest.py`

15. **Clean up dependencies** pyproject.toml
    - Remove or demote `anndata`, `scanpy` to optional (keep for visualization only)
    - Add: `polars[parquet]` explicitly, `sqlalchemy` (optional), `datashader`, `holoviews`
    - Add: `numba` (optional, for faster statistical tests if needed)
    - Recommend: **Keep anndata/scanpy optional but recommended for plots only**

---

### **Relevant Files**

| Path | Current Role | Refactor? | Notes |
|------|--------------|-----------|-------|
| src/epykit/io/parquet_converter.py | **NEW** | Create | Replaces anndata_builder*.py |
| src/epykit/core/parquet_backend.py | **NEW** | Create | Replaces MethylData as primary accessor |
| src/epykit/stats/dmr_processor_chromwise.py | **NEW** | Create | Chromosome-wise DMC orchestration |
| src/epykit/stats/calculate_diff_meth_chromwise.py | **NEW** | Create | Global DMC coordinator (FDR) |
| src/epykit/plot/qc_parquet.py | **NEW** | Create | Parquet-native QC stats |
| src/epykit/plot/datashader_plots.py | **NEW** | Create | Rasterized genome plots |
| bismark.py | ✅ Keep | Minor | Keep format detection, update signature to return Polars |
| generic.py | ✅ Keep | Minor | Keep format detection, update to return Polars |
| sample_sheet.py | Update | Refactor | Add `read_samples_to_parquet()` orchestrator |
| methyldata.py | ❌ Delete/Replace | Refactor | Replace with ParquetMethylStore or minimal wrapper |
| src/epykit/io/anndata_builder*.py | ❌ Delete | Remove | (All 3 files: builder, streaming, chunked) |
| tests.py | Update | Refactor | Change input signatures to per-chrom count arrays (50–150 lines) |
| dmr.py | ✅ Keep | No change | `merge_dmrs()` already array-agnostic |
| glm_vectorized.py | ✅ Keep | No change | Already vectorized |
| tiling.py | Update | Refactor | Adapt to Parquet API; consider pre-aggregation |
| qc.py | Partial | Deprecate | Keep PCA/heatmap logic (on small AnnData), replace scatter with datashader |
| workflow/scripts/*.py | Update | Rewrite | Main orchestration scripts; new API calls |
| pyproject.toml | Update | Refactor | Adjust dependencies, add Parquet/datashader |
| tests/*.py | Update | Refactor | Update test signatures for new functions |

---

### **Verification**

1. **Unit tests** (Phase 5):
   - `test_convert_sample()` — verify Parquet structure for one sample
   - `test_parquet_backend_load_chromosome()` — verify lazy/eager loading, filtering
   - `test_dmr_processor_chromwise()` — compare results to current DMC output (should match exactly)
   - `test_fdr_correction()` — genome-wide FDR computed correctly
   - `test_tiling()` — window aggregation matches old pipeline

2. **Integration test** (End-to-end):
   - Run full pipeline on test data (e.g., 6 samples from test_6samples.py)
   - Compare to current AnnData pipeline:
     - Identical DMC p-values (bit-level, or <1e-10 relative error due to floating point)
     - Identical DMR regions (identical or <1 bp difference due to discretization)
     - Memory usage plots: confirm <500 MB per chromosome
   - Runtime: should be similar or faster (vectorization + Polars streaming benefits)

3. **Memory profiling** (Phase 3–4):
   - Use `memory_profiler` or `tracemalloc` on 100-sample cohort
   - Confirm peak RSS < 2 GB with current-generation laptop hardware (16 GB RAM)
   - Compare to current builder memory envelope

4. **Manual validation**:
   - Run workflow scripts on test data; inspect DMC.tsv, DMR.bed outputs
   - Spot-check Parquet store structure (`tree methyl_store/` output)
   - Run QC plots; confirm PCA and coverage plots render without error

---

### **Decisions**

- **AnnData dependency**: Demote to optional (visualization only). Remove from core pipeline.
- **Backwards compatibility**: NONE — this is a complete switchover. Old AnnData files cannot be read by new pipeline.
- **Parquet schema**: Fixed (chrom, pos, strand, N_meth, N_unmeth, coverage, sample). No versioning necessary initially; change detection happens at write time.
- **Chromosome ordering**: Assume user's input files are sorted or can be sorted. Parquet partitioning is unordered; add `sort_by="chrom"` when reading if genome-order matters.
- **Sample names**: Must be filesystem-safe (alphanumeric, `_`, `-` only). Validation in Phase 1.
- **Statistics**: **Beta-binomial NOT implemented in Phase 1** (scope creep). Use Fisher/GLM/Limma (existing). Beta-binomial can be future enhancement.
- **Missing sites strategy**: **Inner join (only sites in all samples)**. Simplest, matches current "unite(type='intersect')".

---

### **Further Considerations**

1. **Which stats test to prioritize?**
   - Current code uses auto-selection (Fisher if 1 rep/group, GLM if 2+, Limma if any rep).
   - Do you want to keep this? Or expose test choice directly?
   - **Recommendation**: Keep auto-selection; expose via `test="auto" | "fisher" | "glm" | "limma"` parameter.

2. **Batch effect correction in new pipeline?**
   - Current pipeline can use Limma with design matrix including batch term.
   - New pipeline can do the same (extract design matrix, pass to `limma_ebayes_test()`).
   - Ensure workflow scripts accept batch term in metadata.
   - **Recommendation**: Implement in Phase 2 test refactor.

3. **Performance optimization — pre-compute summary statistics?**
   - During conversion (Phase 1), emit a `.stats.parquet` with per-sample/chrom summary (coverage, beta distribution).
   - Use for QC plots without re-scanning.
   - **Recommendation**: Phase 1 enhancement (low priority, but fast to add).

</details> 

Updated memory file [](file:///home/codespace/.vscode-remote/data/User/workspaceStorage/481fd428/GitHub.copilot-chat/memory-tool/memories/ZjFlZDVhMGItM2VjYi00MTExLWJhYWMtNjFiYjgzYzQ4MGQ5/plan.md)

---

## **Summary & Recommendation**

I've explored your codebase in depth and designed a refactoring plan. **Here's the verdict:**

### **Is this manageable? YES — ~70% of code stays, ~30% needs rewriting.**

**Reasons it's manageable:**
1. ✅ **Existing code is clean**: Your stats functions are already vectorized (Fisher, GLM, Limma). They don't need architectural changes—just input shape changes.
2. ✅ **You already use Polars extensively**: The conversion logic you proposed is already partially in place (`read_bismark_coverage`, lazy scans). Minimal new dependencies.
3. ✅ **DMR logic stays the same**: Your `merge_dmrs()` function doesn't care if input is AnnData or Polars. Reusable.
4. ✅ **Tiling is self-contained**: Adapt it to the new API; most logic unchanged.

**What actually changes:**
- **Delete** 3 files: `anndata_builder*.py` (builders you won't need)
- **Create** 6 new files: Parquet converter, backend, chromwise DMC processor, QC plots, datashader, orchestrator
- **Refactor** ~8 files: Pipeline scripts, `tests.py` function signatures (50–150 lines per), tiling, sample_sheet
- **Remove optional**: AnnData from core pipeline (keep for visualization)

### **Timeline**
- **Phase 1 (I/O)**: ~3–5 days (new converter, Parquet backend)
- **Phase 2 (Stats)**: ~2–3 days (chromwise DMC processor, adapt tests.py)
- **Phase 3–4 (Post-processing + Plots)**: ~2–3 days
- **Phase 5 (Testing)**: ~2 days
- **Total**: ~1–1.5 weeks of focused work

### **Next Steps**
1. **Review the plan above** — any requirements I missed or disagreements?
2. **Clarify** if you want to start with Phase 1 (I/O conversion) or if you'd like me to sketch out code structure first
3. **Lock down** the Parquet schema (I used your proposed schema; confirm it works for your data formats)
4. **Identify any experimental features** in current code that should be preserved (e.g., any custom stats tests, specific QC logic)

Is this plan acceptable? Should I proceed to Phase 1 implementation, or would you like refinements first?
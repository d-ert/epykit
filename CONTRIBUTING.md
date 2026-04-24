# Contributing to EpyKit

Thank you for your interest in contributing! This guide covers the development
setup, coding conventions, and the process for submitting pull requests.

---

## Development Setup

### 1. Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- git

### 2. Clone & install

```bash
git clone https://github.com/your-org/EpyKit.git
cd EpyKit

# Install in editable mode with all dev dependencies
uv pip install -e ".[dev]"

# OR with standard pip
pip install -e ".[dev]"
```

### 3. Set up pre-commit hooks

```bash
pre-commit install
```

This installs Ruff (linting + formatting) and mypy (type checking) as
commit hooks that run automatically on every `git commit`.

---

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=pymethyl --cov-report=html

# Run a specific module
uv run pytest tests/test_io.py -v

# Run a specific test
uv run pytest tests/test_stats.py::TestMergeDMRs::test_basic_merge -v
```

Tests use the toy 4-sample / 10-site dataset defined in `tests/conftest.py`
for speed. All fixtures are session-scoped where possible.

---

## Project Structure

```
EpyKit/
├── src/pymethyl/           # Package source (src/ layout)
│   ├── io/                 # Data ingestion
│   │   ├── bismark.py      # Bismark readers
│   │   ├── generic.py      # bedGraph / generic readers
│   │   ├── sample_sheet.py # Multi-sample loader
│   │   └── anndata_builder.py  # AnnData construction + persistence
│   ├── core/               # MethylData wrapper
│   │   └── methyldata.py
│   ├── intervals/          # Genomic interval algebra
│   │   └── tiling.py
│   ├── stats/              # Statistical tests
│   │   ├── tests.py        # Fisher, GLM+LRT, HC0
│   │   └── dmr.py          # DMR merging
│   └── plot/               # Visualisation
│       └── qc.py
├── tests/                  # pytest test suite
├── workflow/               # Snakemake pipeline
│   ├── Snakefile
│   ├── config/config.yaml
│   ├── envs/pymethyl.yaml
│   └── scripts/
├── bioconda/meta.yaml      # Bioconda recipe
├── pyproject.toml
└── .github/workflows/test.yml
```

---

## Coding Conventions

### Style
- **Ruff** for linting and formatting (`ruff check` + `ruff format`)
- Line length: 100 characters
- All public functions must have NumPy-style docstrings

### Type Hints
- All function signatures must be fully typed
- Use `from __future__ import annotations` at the top of each file
- Run `mypy src/pymethyl/` to verify

### AnnData Geometry (Critical!)
> ⚠️ AnnData stores **samples as rows** (obs) and **sites as columns** (var).
> This is the **opposite** of R's methylKit/SummarizedExperiment convention.
> Never violate this — the entire ecosystem (scanpy, scverse) assumes this layout.

---

## Adding Support for a New Aligner

To add support for a new aligner (e.g., bwa-meth, BSMAP, BSBolt):

1. **Add a reader** in `src/pymethyl/io/` (e.g., `bwameth.py`)
2. **Normalise to the standard schema:**
   ```
   chr, start, end, strand, beta, methylated, unmethylated, coverage, context
   ```
3. **Register auto-detection** in `sample_sheet.py::_detect_format()`
4. **Write tests** in `tests/test_io.py`
5. **Update README** table of supported formats

For bwa-meth specifically, the output is a bedGraph-like format. Use
`read_generic_methylation()` with appropriate column mapping parameters
as a starting point.

---

## Statistical Model Notes

### Adding a new test
All test functions in `stats/tests.py` must:
1. Accept a `MethylData` object + `idx_a` / `idx_b` boolean arrays
2. Return `(pvalues: np.ndarray, mean_diff: np.ndarray)` (plus any extras)
3. Handle degenerate cases (all-zero counts) gracefully with `pvalue=1.0`
4. Be registered in `calculate_diff_meth()` via the `test=` parameter

### Overdispersion
The HC0 robust covariance approach is mathematically equivalent to
methylKit's quasi-binomial McCullagh-Nelder correction. Do not replace
this with a true `QuasiBinomial` family until statsmodels natively supports
LRT for quasi-likelihood models.

---

## Pull Request Process

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/add-bwameth-reader
   ```
2. Write code + tests. Ensure `pytest` passes and coverage ≥ 80%.
3. Run the full linting suite:
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   mypy src/pymethyl/
   ```
4. Open a pull request against `main`. The CI will run tests on Python
   3.9, 3.10, 3.11, and 3.12 across Linux, macOS, and Windows.

---

## Releasing a New Version

```bash
# Tag a new version (hatch-vcs derives version from git tags)
git tag v0.2.0
git push origin v0.2.0

# CI will build the wheel and upload to PyPI
# Update bioconda/meta.yaml version + sha256 and submit PR to bioconda-recipes
```

---

## License

By contributing, you agree that your contributions will be licensed under
the [MIT License](LICENSE).

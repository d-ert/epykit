"""
pymethyl.plot.qc
================
Quality control and exploratory data analysis plots.

All functions return Matplotlib ``Figure`` objects (or axes), so they
can be integrated into notebooks or saved to disk::

    fig = pca(adata)
    fig.savefig("pca.png", dpi=150, bbox_inches="tight")

Backend
-------
- **scanpy** for PCA and UMAP computations on AnnData
- **seaborn** for heatmaps and distribution plots
- **matplotlib** as the base rendering engine
- **scipy.stats** for batch-effect association tests
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.figure

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError("anndata is required: pip install anndata") from e

try:
    import scanpy as sc
    _HAS_SCANPY = True
except ImportError:
    _HAS_SCANPY = False
    logger.warning("scanpy not installed. PCA and UMAP plots require scanpy.")


# ---------------------------------------------------------------------------
# Coverage histogram
# ---------------------------------------------------------------------------

def coverage_hist(
    mdata,
    *,
    max_cov: int = 200,
    bins: int = 100,
    figsize: tuple[float, float] | None = None,
    title: str = "Per-sample coverage distribution",
    color: str = "steelblue",
    alpha: float = 0.6,
) -> "matplotlib.figure.Figure":
    """Plot per-sample read coverage distributions.

    Parameters
    ----------
    mdata:
        A ``MethylData`` object or ``AnnData`` with ``layers['coverage']``.
    max_cov:
        Upper limit of x-axis.  Default: 200.
    bins:
        Number of histogram bins.  Default: 100.
    figsize:
        Figure size.  Defaults to ``(4 * n_cols, 3 * n_rows)``.
    title:
        Overall figure title.
    color:
        Bar color.
    alpha:
        Bar transparency.

    Returns
    -------
    matplotlib.figure.Figure
    """
    adata = _get_adata(mdata)
    cov = np.asarray(adata.layers["coverage"])
    n_samples = adata.n_obs
    sample_ids = list(adata.obs_names)

    n_cols = min(4, n_samples)
    n_rows = int(np.ceil(n_samples / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, sid in enumerate(sample_ids):
        ax = axes_flat[i]
        row_cov = cov[i, :]
        covered = row_cov[row_cov > 0]
        ax.hist(
            np.clip(covered, 0, max_cov),
            bins=bins,
            color=color,
            alpha=alpha,
            edgecolor="none",
        )
        mean_cov = float(np.mean(covered)) if len(covered) > 0 else 0
        ax.axvline(mean_cov, color="red", linestyle="--", linewidth=1, label=f"mean={mean_cov:.1f}x")
        ax.set_title(sid, fontsize=9)
        ax.set_xlabel("Coverage depth", fontsize=8)
        ax.set_ylabel("Number of sites", fontsize=8)
        ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(n_samples, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca(
    adata: "ad.AnnData",
    *,
    n_comps: int = 50,
    color_by: str | None = None,
    figsize: tuple[float, float] = (7, 6),
    title: str = "PCA — beta values",
    return_fig: bool = True,
    inplace: bool = True,
) -> "matplotlib.figure.Figure | None":
    """Run PCA on the beta-value matrix and plot PC1 vs PC2.

    Calls ``scanpy.pp.pca()`` and stores embeddings in ``adata.obsm['X_pca']``.

    Parameters
    ----------
    adata:
        AnnData with beta-value matrix in ``X``.
    n_comps:
        Number of principal components to compute.  Default: 50.
    color_by:
        Column in ``adata.obs`` to colour points by.  Default: ``None``
        (uniform colour).
    figsize:
        Figure dimensions.
    title:
        Plot title.
    return_fig:
        If ``True`` (default), return the figure object.
    inplace:
        Store PCA result in ``adata`` in place.  Default: ``True``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _HAS_SCANPY:
        raise ImportError("scanpy is required for PCA: pip install scanpy")

    # Replace NaN with column mean before PCA
    X = np.asarray(adata.X).copy().astype(np.float32)
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    adata_copy = adata.copy()
    adata_copy.X = X

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_comps_actual = min(n_comps, min(adata.n_obs, adata.n_vars) - 1)
        sc.pp.pca(adata_copy, n_comps=n_comps_actual)

    if inplace:
        adata.obsm["X_pca"] = adata_copy.obsm["X_pca"]
        adata.uns["pca"] = adata_copy.uns.get("pca", {})

    pca_coords = adata_copy.obsm["X_pca"]
    variance_ratio = adata_copy.uns.get("pca", {}).get("variance_ratio", None)

    fig, ax = plt.subplots(figsize=figsize)

    if color_by is not None and color_by in adata.obs.columns:
        groups = adata.obs[color_by].astype(str)
        unique_groups = sorted(groups.unique())
        palette = sns.color_palette("Set2", len(unique_groups))
        color_map = dict(zip(unique_groups, palette))
        for grp in unique_groups:
            mask = groups == grp
            ax.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=[color_map[grp]],
                label=grp,
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.legend(title=color_by, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    else:
        ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            s=60,
            alpha=0.85,
            c="steelblue",
            edgecolors="white",
            linewidths=0.5,
        )

    # Label points with sample IDs
    for i, sid in enumerate(adata.obs_names):
        ax.annotate(
            str(sid),
            (pca_coords[i, 0], pca_coords[i, 1]),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    x_var = f" ({variance_ratio[0]*100:.1f}%)" if variance_ratio is not None else ""
    y_var = f" ({variance_ratio[1]*100:.1f}%)" if variance_ratio is not None else ""
    ax.set_xlabel(f"PC1{x_var}", fontsize=11)
    ax.set_ylabel(f"PC2{y_var}", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig if return_fig else None


# ---------------------------------------------------------------------------
# Batch effect detection (assoc_comp)
# ---------------------------------------------------------------------------

def assoc_comp(
    adata: "ad.AnnData",
    *,
    n_comps: int = 10,
    metadata_cols: list[str] | None = None,
    pvalue_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str = "PCA component ~ metadata associations",
) -> tuple["matplotlib.figure.Figure", pd.DataFrame]:
    """Detect batch effects by associating PCA components with metadata.

    This is the Python equivalent of ``methylKit::assocComp()``.

    For each combination of (PC, metadata column):
    - **Categorical** metadata: Kruskal-Wallis H-test
    - **Continuous** metadata: Pearson linear regression (p-value from t-test)

    Parameters
    ----------
    adata:
        AnnData with ``obsm['X_pca']`` (call :func:`pca` first).
        Must have metadata columns in ``obs``.
    n_comps:
        Number of PCA components to test.  Default: 10.
    metadata_cols:
        Columns in ``adata.obs`` to test.  If ``None``, uses all
        non-constant columns.
    pvalue_threshold:
        Significance threshold.  Components below this are highlighted.
    figsize:
        Figure size.  Auto-computed if ``None``.
    title:
        Plot title.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The heatmap figure and a DataFrame of p-values
        (rows = PCs, columns = metadata variables).

    Examples
    --------
    >>> pca(adata)
    >>> fig, pvals = assoc_comp(adata, metadata_cols=["batch", "age", "group"])
    """
    if "X_pca" not in adata.obsm:
        logger.info("PCA not found — running pca() first.")
        pca(adata, n_comps=max(n_comps, 50), return_fig=False)

    pca_coords = adata.obsm["X_pca"]
    n_comps_actual = min(n_comps, pca_coords.shape[1])

    if metadata_cols is None:
        metadata_cols = [
            c for c in adata.obs.columns
            if adata.obs[c].nunique() > 1
        ]

    if not metadata_cols:
        raise ValueError("No variable metadata columns found in adata.obs.")

    # Build p-value matrix
    pval_records = {}
    for col in metadata_cols:
        col_vals = adata.obs[col].values
        is_categorical = (
            col_vals.dtype.kind in ("O", "U", "S")
            or pd.api.types.is_categorical_dtype(col_vals)
            or pd.Series(col_vals).nunique() <= 10
        )
        row = []
        for pc_i in range(n_comps_actual):
            pc_scores = pca_coords[:, pc_i]
            if is_categorical:
                groups = [pc_scores[col_vals == g] for g in np.unique(col_vals)]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    row.append(1.0)
                else:
                    _, p = scipy_stats.kruskal(*groups)
                    row.append(float(p))
            else:
                try:
                    vals_num = col_vals.astype(float)
                    valid = ~(np.isnan(vals_num) | np.isnan(pc_scores))
                    if valid.sum() < 3:
                        row.append(1.0)
                    else:
                        _, _, _, p, _ = scipy_stats.linregress(
                            vals_num[valid], pc_scores[valid]
                        )
                        row.append(float(p))
                except Exception:
                    row.append(1.0)
        pval_records[col] = row

    pc_labels = [f"PC{i+1}" for i in range(n_comps_actual)]
    pval_df = pd.DataFrame(pval_records, index=pc_labels)

    # Log-transform p-values for display: -log10(p)
    log_pvals = -np.log10(pval_df.values.clip(1e-300, 1.0))
    log_pval_df = pd.DataFrame(log_pvals, index=pc_labels, columns=list(pval_df.columns))

    # --- Plot ---
    if figsize is None:
        figsize = (max(6, len(metadata_cols) * 1.2), max(4, n_comps_actual * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sig_threshold = -np.log10(pvalue_threshold)

    sns.heatmap(
        log_pval_df,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "-log₁₀(p-value)"},
        vmin=0,
        vmax=max(sig_threshold * 3, log_pvals.max() + 1),
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Metadata variable", fontsize=10)
    ax.set_ylabel("Principal component", fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    n_sig = int((pval_df.values < pvalue_threshold).sum())
    logger.info(
        "assoc_comp: %d significant (PC, metadata) associations found (p<%.3f)",
        n_sig, pvalue_threshold,
    )

    return fig, pval_df


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def umap(
    adata: "ad.AnnData",
    *,
    color_by: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    figsize: tuple[float, float] = (7, 6),
    title: str = "UMAP — beta values",
) -> "matplotlib.figure.Figure":
    """Compute and plot UMAP dimensionality reduction.

    Requires scanpy.  Calls ``scanpy.pp.neighbors()`` then
    ``scanpy.tl.umap()``.  PCA must have been computed first (call
    :func:`pca`).

    Parameters
    ----------
    adata:
        AnnData with ``obsm['X_pca']``.
    color_by:
        Column in ``adata.obs`` to colour points by.
    n_neighbors:
        Number of neighbors for the kNN graph.  Default: 15.
    min_dist:
        UMAP minimum distance parameter.  Default: 0.3.
    figsize:
        Figure dimensions.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _HAS_SCANPY:
        raise ImportError("scanpy is required for UMAP: pip install scanpy")

    if "X_pca" not in adata.obsm:
        logger.info("PCA not found — running pca() with 50 components.")
        pca(adata, n_comps=50, return_fig=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X_pca")
        sc.tl.umap(adata, min_dist=min_dist)

    umap_coords = adata.obsm["X_umap"]

    fig, ax = plt.subplots(figsize=figsize)

    if color_by is not None and color_by in adata.obs.columns:
        groups = adata.obs[color_by].astype(str)
        unique_groups = sorted(groups.unique())
        palette = sns.color_palette("Set2", len(unique_groups))
        for i, grp in enumerate(unique_groups):
            mask = groups == grp
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[palette[i]],
                label=grp,
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.legend(title=color_by, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    else:
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            s=60,
            alpha=0.85,
            c="steelblue",
            edgecolors="white",
            linewidths=0.5,
        )

    for i, sid in enumerate(adata.obs_names):
        ax.annotate(
            str(sid),
            (umap_coords[i, 0], umap_coords[i, 1]),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("UMAP1", fontsize=11)
    ax.set_ylabel("UMAP2", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Sample correlation heatmap
# ---------------------------------------------------------------------------

def sample_correlation(
    adata: "ad.AnnData",
    *,
    method: str = "pearson",
    figsize: tuple[float, float] | None = None,
    title: str = "Sample correlation (beta values)",
    color_by: str | None = None,
    cmap: str = "RdBu_r",
    vmin: float = 0.8,
    vmax: float = 1.0,
) -> "matplotlib.figure.Figure":
    """Plot a sample-sample Pearson correlation heatmap.

    Uses ``seaborn.clustermap`` with hierarchical clustering.

    Parameters
    ----------
    adata:
        AnnData with beta-value matrix in ``X``.
    method:
        Correlation method: ``"pearson"`` or ``"spearman"``.
    figsize:
        Figure dimensions.  Auto-computed if ``None``.
    title:
        Figure title.
    color_by:
        Column in ``adata.obs`` to add as a colour sidebar.
    cmap:
        Colormap.  Default: ``"RdBu_r"``.
    vmin, vmax:
        Colour scale limits.  Default: 0.8–1.0.

    Returns
    -------
    matplotlib.figure.Figure
    """
    X = np.asarray(adata.X).copy().astype(np.float64)
    # Impute NaN with row mean
    for i in range(X.shape[0]):
        row = X[i, :]
        nan_mask = np.isnan(row)
        if nan_mask.any():
            X[i, nan_mask] = np.nanmean(row) if not np.all(nan_mask) else 0.0

    sample_ids = list(adata.obs_names)
    n = len(sample_ids)

    if figsize is None:
        figsize = (max(6, n * 0.6), max(5, n * 0.6))

    if method == "pearson":
        corr_matrix = np.corrcoef(X)
    elif method == "spearman":
        corr_matrix = np.array([
            [scipy_stats.spearmanr(X[i, :], X[j, :])[0] for j in range(n)]
            for i in range(n)
        ])
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pearson' or 'spearman'.")

    corr_df = pd.DataFrame(corr_matrix, index=sample_ids, columns=sample_ids)

    # Row colour annotations
    row_colors = None
    if color_by is not None and color_by in adata.obs.columns:
        groups = adata.obs[color_by].astype(str)
        unique_groups = sorted(groups.unique())
        palette = dict(zip(unique_groups, sns.color_palette("Set2", len(unique_groups))))
        row_colors = groups.map(palette)

    g = sns.clustermap(
        corr_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        row_colors=row_colors,
        col_colors=row_colors,
        annot=n <= 20,
        fmt=".2f" if n <= 20 else "",
        linewidths=0.5 if n <= 30 else 0.0,
        xticklabels=True,
        yticklabels=True,
    )
    g.fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return g.fig


# ---------------------------------------------------------------------------
# Methylation distribution
# ---------------------------------------------------------------------------

def methylation_distribution(
    mdata,
    *,
    plot_type: str = "violin",
    figsize: tuple[float, float] | None = None,
    color_by: str | None = None,
    title: str = "Per-sample methylation distribution",
) -> "matplotlib.figure.Figure":
    """Plot per-sample beta-value distributions.

    Parameters
    ----------
    mdata:
        MethylData or AnnData object.
    plot_type:
        ``"violin"`` (default), ``"box"``, or ``"hist"``.
    figsize:
        Figure size.
    color_by:
        Column in obs for colour grouping.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    adata = _get_adata(mdata)
    X = np.asarray(adata.X)
    sample_ids = list(adata.obs_names)
    n = len(sample_ids)

    if figsize is None:
        figsize = (max(6, n * 0.7), 5)

    fig, ax = plt.subplots(figsize=figsize)

    # Collect per-sample beta values (non-NaN only)
    data_list = []
    labels = []
    for i, sid in enumerate(sample_ids):
        row = X[i, :]
        valid = row[~np.isnan(row)]
        data_list.append(valid)
        labels.append(str(sid))

    if plot_type == "violin":
        parts = ax.violinplot(
            data_list,
            positions=range(n),
            showmedians=True,
            showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    elif plot_type == "box":
        ax.boxplot(
            data_list,
            labels=labels,
            patch_artist=True,
            boxprops={"alpha": 0.7},
        )
        plt.xticks(rotation=30, ha="right", fontsize=9)

    elif plot_type == "hist":
        palette = sns.color_palette("husl", n)
        for i, (vals, sid) in enumerate(zip(data_list, labels)):
            ax.hist(vals, bins=50, alpha=0.5, color=palette[i], label=sid, density=True)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        raise ValueError(f"Unknown plot_type: '{plot_type}'. Use 'violin', 'box', or 'hist'.")

    ax.set_xlabel("Sample", fontsize=11)
    ax.set_ylabel("Beta value (methylation %)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_adata(obj) -> "ad.AnnData":
    """Extract AnnData from either MethylData or AnnData."""
    if hasattr(obj, "adata"):
        return obj.adata
    return obj

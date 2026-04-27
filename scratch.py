import epykit

"""
adata = epykit.io.read_samples(
    'small_samplesheet.csv',
    engine='duckdb',
    min_coverage=10,
    output='zarr',
    out_path='cohort.zarr'
)

"""


# Option A (recommended): epykit helper
adata = epykit.io.load("cohort.zarr")

from epykit.core import MethylData

mdata = MethylData(adata)
print(mdata)

from epykit import plot as epyplot

# coverage distributions
fig = epyplot.coverage_hist(mdata, max_cov=200)
fig.savefig("coverage_hist.png", dpi=150, bbox_inches="tight")

# beta distributions
fig = epyplot.methylation_distribution(mdata) 
fig.savefig("meth_dist.png", dpi=150, bbox_inches="tight")

# PCA / sample correlation
fig = epyplot.pca(mdata.adata, color_by="group")  # uses adata.obs['group']
fig.savefig("pca.png", dpi=150, bbox_inches="tight")

fig = epyplot.sample_correlation(mdata.adata)
fig.savefig("sample_corr.png", dpi=150, bbox_inches="tight")


mdata = (
    mdata
    .filter_coverage(min_cov=10, max_cov=1000)   # tune to your dataset
    .subset_context("CpG")                      # if context exists
    .unite(type="intersect")                    # keep sites covered in all samples
)

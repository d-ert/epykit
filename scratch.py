import epykit


adata = epykit.io.read_samples(
    'small_samplesheet.csv',
    engine='duckdb',
    min_coverage=10,
    output='zarr',
    out_path='cohort.zarr'
)


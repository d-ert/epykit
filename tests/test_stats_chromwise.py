from __future__ import annotations

import pandas as pd


class TestChromwiseDiffMeth:
    def test_process_chromosome(self, sample_sheet_dir, tmp_path):
        from epykit.core import ParquetMethylStore
        from epykit.io import read_samples_to_parquet
        from epykit.stats.dmr_processor_chromwise import process_chromosome

        sheet = sample_sheet_dir / "sample_sheet.csv"
        out_dir = tmp_path / "methylstore"

        read_samples_to_parquet(sheet, out_dir, n_workers=1, min_coverage=1, chunksize=10_000)
        store = ParquetMethylStore(out_dir)
        metadata = pd.read_csv(sheet).set_index("sample_id")

        result = process_chromosome(
            store,
            "chr1",
            samples=metadata.index.tolist(),
            group_col="group",
            metadata=metadata,
            test="fisher",
        )

        assert result.height > 0
        for col in ("chr", "start", "end", "pvalue", "mean_diff"):
            assert col in result.columns

    def test_calculate_diff_meth_chromwise(self, sample_sheet_dir, tmp_path):
        from epykit.core import ParquetMethylStore
        from epykit.io import read_samples_to_parquet
        from epykit.stats.calculate_diff_meth_chromwise import calculate_diff_meth_chromwise

        sheet = sample_sheet_dir / "sample_sheet.csv"
        out_dir = tmp_path / "methylstore"
        out_file = tmp_path / "dmc.tsv"

        read_samples_to_parquet(sheet, out_dir, n_workers=1, min_coverage=1, chunksize=10_000)
        store = ParquetMethylStore(out_dir)
        metadata = pd.read_csv(sheet).set_index("sample_id")

        result = calculate_diff_meth_chromwise(
            store,
            samples=metadata.index.tolist(),
            group_col="group",
            metadata=metadata,
            test="fisher",
            output_path=out_file,
            n_threads=1,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "qvalue" in result.columns
        assert out_file.exists()
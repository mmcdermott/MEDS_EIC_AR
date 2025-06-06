from pathlib import Path

import pyarrow.parquet as pq
from meds_torchdata import MEDSPytorchDataset

from conftest import run_process_data


def test_process_data_runs(preprocessed_dataset: Path):
    out_files = list(preprocessed_dataset.rglob("*.*"))
    assert len(out_files) > 0


def test_process_dataset_correct(pytorch_dataset: MEDSPytorchDataset):
    assert len(pytorch_dataset) > 0


def test_process_with_reshard_correct(preprocessed_dataset_with_reshard: Path):
    out_files = list(preprocessed_dataset_with_reshard.rglob("*.*"))
    assert len(out_files) > 0


def test_process_with_custom_bins(simple_static_MEDS: Path, tmp_path, monkeypatch):
    bins_fp = tmp_path / "bins.yaml"
    bins_fp.write_text("{HR: {low: 100, high: 110}}")
    monkeypatch.setenv("NUMERIC_QUANTILES_FP", str(bins_fp))
    output = run_process_data(simple_static_MEDS, tmp_path)
    codes = pq.read_table(output / "metadata" / "codes.parquet").column("code").to_pylist()
    hr_codes = [c for c in codes if c.startswith("HR")]
    assert set(hr_codes) == {
        "HR//value_[-inf,100.0)",
        "HR//value_[100.0,110.0)",
        "HR//value_[110.0,inf)",
    }


def test_process_without_numeric(simple_static_MEDS: Path, tmp_path, monkeypatch):
    monkeypatch.setenv("INCLUDE_NUMERIC_VALUES", "0")
    output = run_process_data(simple_static_MEDS, tmp_path)
    codes = pq.read_table(output / "metadata" / "codes.parquet").column("code").to_pylist()
    assert not any("//value_" in c for c in codes)

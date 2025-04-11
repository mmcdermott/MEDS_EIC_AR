from pathlib import Path

from meds_torchdata import MEDSPytorchDataset


def test_process_data_runs(preprocessed_dataset: Path):
    out_files = list(preprocessed_dataset.rglob("*.*"))
    assert len(out_files) > 0


def test_process_dataset_correct(pytorch_dataset: MEDSPytorchDataset):
    assert len(pytorch_dataset) > 0

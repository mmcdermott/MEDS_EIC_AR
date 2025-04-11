"""Test set-up and fixtures code."""

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from torch.utils.data import DataLoader


@pytest.fixture(scope="session")
def preprocessed_dataset(simple_static_MEDS: Path) -> Path:
    """Fixture to create a preprocessed dataset."""

    with tempfile.TemporaryDirectory() as test_root:
        test_root = Path(test_root)

        input_dir = simple_static_MEDS
        interemediate_dir = test_root / "intermediate"
        output_dir = test_root / "output"

        cmd = [
            "MEICAR_process_data",
            f"input_dir={input_dir!s}",
            f"intermediate_dir={interemediate_dir!s}",
            f"output_dir={output_dir!s}",
            "do_demo=True",
        ]

        out = subprocess.run(cmd, capture_output=True, check=False)

        err_lines = [
            "Command failed:",
            "Stdout:",
            out.stdout.decode(),
            "Stderr:",
            out.stderr.decode(),
        ]

        if out.returncode != 0:
            raise ValueError("\n".join([*err_lines, f"Return code: {out.returncode}"]))

        yield output_dir


@pytest.fixture(scope="session")
def pytorch_dataset(preprocessed_dataset: Path) -> MEDSPytorchDataset:
    """Fixture to create a PyTorch dataset."""
    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=preprocessed_dataset,
        max_seq_len=10,
    )

    return MEDSPytorchDataset(config, split="train")


@pytest.fixture(scope="session")
def sample_batch(pytorch_dataset: MEDSPytorchDataset) -> MEDSTorchBatch:
    """Fixture to create a sample batch."""
    dataloader = DataLoader(pytorch_dataset, batch_size=2, shuffle=False, collate_fn=pytorch_dataset.collate)
    return next(iter(dataloader))


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    simple_static_MEDS: Path,
    simple_static_MEDS_dataset_with_task: Path,
    sample_batch: MEDSTorchBatch,
    preprocessed_dataset: Path,
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
            "preprocessed_dataset": preprocessed_dataset,
            "sample_batch": sample_batch,
        }
    )

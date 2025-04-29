"""Test set-up and fixtures code."""

import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from meds_testing_helpers.dataset import MEDSDataset
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from torch.utils.data import DataLoader

from MEDS_EIC_AR.model.model import Model
from MEDS_EIC_AR.training.module import MEICARModule


@pytest.fixture(scope="session")
def preprocessed_dataset_with_reshard(simple_static_MEDS: Path) -> Path:
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
            "do_reshard=True",
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
def preprocessed_dataset_with_task(
    preprocessed_dataset: Path,
    simple_static_MEDS_dataset_with_task: Path,
) -> tuple[Path, Path, str]:
    D = MEDSDataset(root_dir=simple_static_MEDS_dataset_with_task)

    if len(D.task_names) != 1:  # pragma: no cover
        raise ValueError("Expected only one task in the dataset.")

    yield preprocessed_dataset, D.task_root_dir, D.task_names[0]


@pytest.fixture(scope="session")
def dataset_config(preprocessed_dataset: Path) -> MEDSTorchDataConfig:
    """Fixture to create a dataset configuration."""
    return MEDSTorchDataConfig(tensorized_cohort_dir=preprocessed_dataset, max_seq_len=10)


@pytest.fixture(scope="session")
def dataset_config_with_task(preprocessed_dataset_with_task: tuple[Path, Path, str]) -> MEDSTorchDataConfig:
    """Fixture to create a dataset configuration."""
    cohort_dir, tasks_dir, task_name = preprocessed_dataset_with_task
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=cohort_dir,
        max_seq_len=10,
        task_labels_dir=(tasks_dir / task_name),
        seq_sampling_strategy="to_end",
    )


@pytest.fixture(scope="session")
def pytorch_dataset(dataset_config: MEDSTorchDataConfig) -> MEDSPytorchDataset:
    """Fixture to create a PyTorch dataset."""
    return MEDSPytorchDataset(dataset_config, split="train")


@pytest.fixture(scope="session")
def pytorch_dataset_with_task(dataset_config_with_task: MEDSTorchDataConfig) -> MEDSPytorchDataset:
    """Fixture to create a PyTorch dataset with task labels."""
    return MEDSPytorchDataset(dataset_config_with_task, split="train")


@pytest.fixture(scope="session")
def sample_batch(pytorch_dataset: MEDSPytorchDataset) -> MEDSTorchBatch:
    """Fixture to create a sample batch."""
    dataloader = DataLoader(pytorch_dataset, batch_size=2, shuffle=False, collate_fn=pytorch_dataset.collate)
    return next(iter(dataloader))


@pytest.fixture(scope="session")
def pretrained_model(preprocessed_dataset: Path) -> Path:
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)

        cmd = [
            "MEICAR_pretrain",
            "--config-name=_demo_pretrain",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={preprocessed_dataset!s}",
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
def pretrained_GPT_model(pretrained_model: Path) -> Model:
    """Returns the HF model backbone of the pre-trained MEICAR Lightning Module."""

    ckpt_path = pretrained_model / "best_model.ckpt"
    if not ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")

    module = MEICARModule.load_from_checkpoint(ckpt_path)
    return module.model


@pytest.fixture(scope="session")
def generated_trajectories(
    pretrained_model: Path, preprocessed_dataset_with_task: tuple[Path, Path, str]
) -> Path:
    tensorized_cohort_dir, task_root_dir, task_name = preprocessed_dataset_with_task
    model_initialization_dir = pretrained_model

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)

        cmd = [
            "MEICAR_generate_trajectories",
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={model_initialization_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir!s}",
            f"datamodule.config.task_labels_dir={(task_root_dir / task_name)!s}",
            "datamodule.batch_size=2",
            "trainer=demo",
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


@contextmanager
def print_warnings(caplog: pytest.LogCaptureFixture):
    """Captures all logged warnings within this context block and prints them upon exit.

    This is useful in doctests, where you want to show printed outputs for documentation and testing purposes.
    """

    n_current_records = len(caplog.records)

    with caplog.at_level("WARNING"):
        yield
    # Print all captured warnings upon exit
    for record in caplog.records[n_current_records:]:
        print(f"Warning: {record.getMessage()}")


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    simple_static_MEDS: Path,
    simple_static_MEDS_dataset_with_task: Path,
    sample_batch: MEDSTorchBatch,
    preprocessed_dataset: Path,
    dataset_config: MEDSTorchDataConfig,
    pretrained_GPT_model: Model,
    pytorch_dataset: MEDSPytorchDataset,
    pytorch_dataset_with_task: MEDSPytorchDataset,
):
    doctest_namespace.update(
        {
            "print_warnings": partial(print_warnings, caplog),
            "patch": patch,
            "MagicMock": MagicMock,
            "Mock": Mock,
            "datetime": datetime,
            "tempfile": tempfile,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
            "preprocessed_dataset": preprocessed_dataset,
            "sample_batch": sample_batch,
            "dataset_config": dataset_config,
            "pretrained_GPT_model": pretrained_GPT_model,
            "pytorch_dataset": pytorch_dataset,
            "pytorch_dataset_with_task": pytorch_dataset_with_task,
        }
    )

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


def run_and_check(cmd: list[str]) -> dict[str, str]:
    """Runs a command and checks the output.

    Args:
        cmd: The command to run. This should be a list of strings, which will be run through subprocess.run in
            shell=False mode.

    Raises:
        ValueError: If the command fails.

    Returns:
        A dictionary with the keys "stdout" and "stderr", containing the output of the command.
    """

    cmd_out = subprocess.run(cmd, capture_output=True, check=False)

    out = {"stdout": cmd_out.stdout.decode(), "stderr": cmd_out.stderr.decode()}

    if cmd_out.returncode == 0:
        return

    err = [f"Command yielded code {cmd_out.returncode}", "Stdout:", out["stdout"], "Stderr:", out["stderr"]]
    raise ValueError("\n".join(err))


def run_process_data(input_dir: Path, root_dir: Path, do_demo: bool = True, do_reshard: bool = False) -> Path:
    """Runs the pre-process code and returns the output directory.

    Args:
        input_dir: The input directory to process.
        root_dir: This is the root dir of the output data, and the intermediate dir is set to
            `root_dir / "intermediate"` and the output dir is set to `root_dir / "output"`.
        do_demo: Whether to run the demo mode.
        do_reshard: Whether to run the resharding step.

    Returns:
        The output directory (`root_dir / "output"`).

    Raises:
        ValueError: If the command fails.
    """

    intermediate_dir = root_dir / "intermediate"
    output_dir = root_dir / "output"

    run_and_check(
        [
            "MEICAR_process_data",
            f"input_dir={input_dir!s}",
            f"intermediate_dir={intermediate_dir!s}",
            f"output_dir={output_dir!s}",
            f"do_demo={do_demo}",
            f"do_reshard={do_reshard}",
        ]
    )

    return output_dir


@pytest.fixture(scope="session")
def preprocessed_dataset_with_reshard(
    simple_static_MEDS: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Fixture to create a preprocessed dataset."""

    test_root = tmp_path_factory.mktemp("preprocessed_dataset_with_reshard")

    return run_process_data(simple_static_MEDS, test_root, do_reshard=True)


@pytest.fixture(scope="session")
def preprocessed_dataset(simple_static_MEDS: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture to create a preprocessed dataset."""

    test_root = tmp_path_factory.mktemp("preprocessed_dataset")

    return run_process_data(simple_static_MEDS, test_root)


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
        include_window_last_observed_in_schema=True,
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
    return list(dataloader)[1]


@pytest.fixture(scope="session")
def pretrained_model(preprocessed_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("pretrained_model")

    run_and_check(
        [
            "MEICAR_pretrain",
            "--config-name=_demo_pretrain",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={preprocessed_dataset!s}",
        ]
    )

    return output_dir


@pytest.fixture(scope="session")
def pretrained_module(pretrained_model: Path) -> MEICARModule:
    """Returns the pre-trained MEICAR Lightning Module."""

    ckpt_path = pretrained_model / "best_model.ckpt"
    if not ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")

    return MEICARModule.load_from_checkpoint(ckpt_path)


@pytest.fixture(scope="session")
def pretrained_GPT_model(pretrained_module: MEICARModule) -> Model:
    """Returns the HF model backbone of the pre-trained MEICAR Lightning Module."""

    return pretrained_module.model


@pytest.fixture(scope="session")
def generated_trajectories(
    pretrained_model: Path,
    preprocessed_dataset_with_task: tuple[Path, Path, str],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    tensorized_cohort_dir, task_root_dir, task_name = preprocessed_dataset_with_task
    model_initialization_dir = pretrained_model

    output_dir = tmp_path_factory.mktemp("generated_trajectories")

    run_and_check(
        [
            "MEICAR_generate_trajectories",
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={model_initialization_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir!s}",
            f"datamodule.config.task_labels_dir={(task_root_dir / task_name)!s}",
            "datamodule.batch_size=2",
            "trainer=demo",
        ]
    )

    return output_dir


@pytest.fixture(scope="session")
def generated_trajectories_rolling(
    pretrained_model: Path,
    preprocessed_dataset_with_task: tuple[Path, Path, str],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Fixture that drives MEICAR_generate_trajectories through the rolling (sliding-window) path.

    Sets ``rolling_generation.max_new_tokens`` to a value larger than the model's per-chunk capacity so the
    internal loop must cross at least one chunk boundary. This exercises the end-to-end wiring from
    ``_generate_trajectories.yaml`` → ``M.generation_kwargs`` → ``Model.generate`` → ``_rolling_generate``.
    """

    tensorized_cohort_dir, task_root_dir, task_name = preprocessed_dataset_with_task
    model_initialization_dir = pretrained_model

    output_dir = tmp_path_factory.mktemp("generated_trajectories_rolling")

    run_and_check(
        [
            "MEICAR_generate_trajectories",
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={model_initialization_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir!s}",
            f"datamodule.config.task_labels_dir={(task_root_dir / task_name)!s}",
            "datamodule.batch_size=2",
            "trainer=demo",
            # Set strictly above ``2 * max_seq_len`` (= 2 * 20 = 40 in ``_demo_pretrain``) so the
            # rolling loop is guaranteed to iterate across several sliding chunks regardless of
            # per-subject input length — not just one boundary crossing. The end-to-end test
            # downstream uses the gap vs. the non-rolling fixture to prove the rolling path
            # actually produced output the single-chunk path could not.
            "rolling_generation.max_new_tokens=50",
        ]
    )

    return output_dir


# ---------------------------------------------------------------------------
# Grammar-dataset fixtures (issue #105) — pretrain + generate on a synthetic
# grammar dataset via the full CLI pipeline so we can validate generation
# correctness end-to-end, not just plumbing.
# ---------------------------------------------------------------------------


# Training-hyperparameter overrides we pass to the grammar pretrain CLI. Centralized here so all
# grammar fixtures agree on them — the CLI test file inlines references to these for its own
# assertions (e.g. "generation output should not be wider than this max_seq_len"). See issue #105.
_GRAMMAR_MAX_SEQ_LEN = 16
_GRAMMAR_MODEL_HEADS = 2
_GRAMMAR_MODEL_HEAD_DIM = 16
_GRAMMAR_PRETRAIN_EPOCHS = 80
_GRAMMAR_PRETRAIN_BATCH_SIZE = 8


@pytest.fixture(scope="session")
def grammar_raw_meds(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build a raw-MEDS directory from the synthetic grammar + a task-labels directory.

    Returns a tuple ``(meds_root, task_labels_dir)`` that matches the shape of
    ``preprocessed_dataset_with_task``: the MEDS root has ``data/`` and ``metadata/`` subdirs
    ready for ``MEICAR_process_data``, and the task-labels dir has per-split parquets keyed by
    subject with a single prediction time per subject.
    """
    from tests._grammar_meds import build_grammar_meds_dataset

    root = tmp_path_factory.mktemp("grammar_raw_meds")
    meds_dir = root / "meds"
    task_dir = root / "task_labels"
    build_grammar_meds_dataset(meds_dir, task_labels_dir=task_dir)
    return meds_dir, task_dir


@pytest.fixture(scope="session")
def grammar_preprocessed(
    grammar_raw_meds: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Runs ``MEICAR_process_data`` on the grammar raw MEDS dir and returns the tensorized dir."""
    meds_dir, _ = grammar_raw_meds
    return run_process_data(meds_dir, tmp_path_factory.mktemp("grammar_preprocessed"))


@pytest.fixture(scope="session")
def grammar_pretrained(grammar_preprocessed: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Runs ``MEICAR_pretrain`` on the grammar preprocessed dataset.

    Uses a bigger model than the default demo preset (hidden_size=32 via 2 heads x head_dim=16)
    and a meaningful training budget (80 epochs x a few batches each) so the model actually has
    a chance to learn the grammar patterns. Disables early-stopping for the same reason — the
    demo patience=3 kicks in after only a handful of validation checks against a noisy baseline.

    Expected runtime: roughly a minute on a laptop CPU. This is the bulk of the grammar
    integration test cost; everything downstream is cheap by comparison.
    """
    output_dir = tmp_path_factory.mktemp("grammar_pretrained")
    run_and_check(
        [
            "MEICAR_pretrain",
            "--config-name=_demo_pretrain",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.batch_size={_GRAMMAR_PRETRAIN_BATCH_SIZE}",
            f"trainer.max_epochs={_GRAMMAR_PRETRAIN_EPOCHS}",
            "trainer.overfit_batches=0",
            "trainer.callbacks.early_stopping.patience=100000",
            f"max_seq_len={_GRAMMAR_MAX_SEQ_LEN}",
            f"lightning_module.model.gpt_kwargs.num_attention_heads={_GRAMMAR_MODEL_HEADS}",
            f"lightning_module.model.gpt_kwargs.attention_head_dim={_GRAMMAR_MODEL_HEAD_DIM}",
        ]
    )
    return output_dir


@pytest.fixture(scope="session")
def grammar_generated_trajectories(
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Runs ``MEICAR_generate_trajectories`` on the grammar-pretrained model.

    Uses a larger ``N_trajectories_per_task_sample=8`` (vs. the demo default of 2) and a
    ``batch_size=3`` that is deliberately *not* a multiple of N, so the interleaving /demux code
    path from #89 is stressed in its cross-sample-group configuration. Rolling generation is
    enabled with a modest budget so we also exercise the sliding-window path end-to-end.
    """
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("grammar_generated_trajectories")
    run_and_check(
        [
            "MEICAR_generate_trajectories",
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={grammar_pretrained!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.config.task_labels_dir={task_dir!s}",
            "datamodule.batch_size=3",
            "trainer=demo",
            "inference.N_trajectories_per_task_sample=8",
            "rolling_generation.max_new_tokens=30",
        ]
    )
    return output_dir


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
    pretrained_model: Path,
    pretrained_GPT_model: Model,
    pretrained_module: MEICARModule,
    pytorch_dataset: MEDSPytorchDataset,
    pytorch_dataset_with_task: MEDSPytorchDataset,
):
    doctest_namespace.update(
        {
            "print_warnings": partial(print_warnings, caplog),
            "patch": patch,
            "MagicMock": MagicMock,
            "Path": Path,
            "Mock": Mock,
            "datetime": datetime,
            "tempfile": tempfile,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
            "preprocessed_dataset": preprocessed_dataset,
            "sample_batch": sample_batch,
            "dataset_config": dataset_config,
            "pretrained_model": pretrained_model,
            "pretrained_GPT_model": pretrained_GPT_model,
            "pretrained_module": pretrained_module,
            "pytorch_dataset": pytorch_dataset,
            "pytorch_dataset_with_task": pytorch_dataset_with_task,
        }
    )

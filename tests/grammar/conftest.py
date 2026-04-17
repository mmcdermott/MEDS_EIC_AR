"""Pytest options + fixtures for the grammar-based end-to-end CLI test (issue #105).

All knobs that control fixture cost (subjects per split, model width, epochs, etc.) are exposed
as pytest CLI options so they can be overridden ad-hoc for debugging without editing code — e.g.
``pytest tests/grammar/test_cli.py --grammar-pretrain-epochs=800 --grammar-n-held-out=32``.
Defaults below are the values the PR thread calibrated against to produce unambiguous
distributional + strict grammar-validity signal on the fixture's trained model.

All fixtures are session-scoped: the pretrain is expensive and both the sampling and greedy
generate paths share it. Multi-fixture runs pay the cost once.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from meds import held_out_split, train_split, tuning_split

if TYPE_CHECKING:
    from pathlib import Path

from tests.grammar._grammar import GrammarFSM
from tests.grammar._meds import (
    CODE_TO_TOKEN,
    GRAMMAR_CODES,
    GRAMMAR_TASK_NAME,
    build_grammar_meds_dataset,
    grammar_tokens_from_output_df,
    prompt_grammar_tokens_by_subject,
)

__all__ = [  # re-export grammar helpers for test files
    "CODE_TO_TOKEN",
    "GRAMMAR_CODES",
    "GRAMMAR_TASK_NAME",
    "GrammarFSM",
    "grammar_tokens_from_output_df",
    "held_out_split",
    "prompt_grammar_tokens_by_subject",
    "train_split",
    "tuning_split",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register grammar-fixture knobs as pytest CLI options.

    These were module-level ``GRAMMAR_*`` constants in an earlier revision; exposing them as
    pytest options makes it trivial to sweep them for test-flakiness debugging or to scale the
    fixture up/down on different machines without a code edit.
    """
    group = parser.getgroup("grammar", "End-to-end grammar-test fixture controls (see tests/grammar/)")
    group.addoption("--grammar-max-seq-len", type=int, default=16)
    group.addoption("--grammar-model-heads", type=int, default=4)
    group.addoption("--grammar-model-head-dim", type=int, default=32)
    group.addoption("--grammar-pretrain-epochs", type=int, default=400)
    group.addoption("--grammar-pretrain-batch-size", type=int, default=32)
    group.addoption("--grammar-n-trajectories", type=int, default=8)
    group.addoption(
        "--grammar-generate-batch-size",
        type=int,
        default=3,
        help=(
            "Deliberately not a multiple of --grammar-n-trajectories so the interleave/demux "
            "code path from PR #103 is stressed in its cross-sample-group configuration."
        ),
    )
    group.addoption(
        "--grammar-rolling-max-new-tokens",
        type=int,
        default=30,
        help=(
            "Must be > --grammar-max-seq-len so the rolling (sliding-window) generation loop "
            "is exercised by at least one sample."
        ),
    )
    group.addoption("--grammar-n-train", type=int, default=256)
    group.addoption("--grammar-n-tuning", type=int, default=16)
    group.addoption("--grammar-n-held-out", type=int, default=16)


# -- Scalar knob fixtures ------------------------------------------------
# One fixture per CLI option; kept session-scoped because the value never varies within a
# session. Tests and builder fixtures consume these by name.


@pytest.fixture(scope="session")
def grammar_max_seq_len(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-max-seq-len")


@pytest.fixture(scope="session")
def grammar_model_heads(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-model-heads")


@pytest.fixture(scope="session")
def grammar_model_head_dim(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-model-head-dim")


@pytest.fixture(scope="session")
def grammar_pretrain_epochs(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-pretrain-epochs")


@pytest.fixture(scope="session")
def grammar_pretrain_batch_size(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-pretrain-batch-size")


@pytest.fixture(scope="session")
def grammar_n_trajectories(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-n-trajectories")


@pytest.fixture(scope="session")
def grammar_generate_batch_size(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-generate-batch-size")


@pytest.fixture(scope="session")
def grammar_rolling_max_new_tokens(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-rolling-max-new-tokens")


@pytest.fixture(scope="session")
def grammar_n_train(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-n-train")


@pytest.fixture(scope="session")
def grammar_n_tuning(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-n-tuning")


@pytest.fixture(scope="session")
def grammar_n_held_out(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--grammar-n-held-out")


# -- Pipeline fixtures ---------------------------------------------------


def _run_and_check(cmd: list[str]) -> None:
    """Small helper: run ``cmd`` via subprocess and raise with captured output on non-zero."""
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError(
            "\n".join(
                [
                    f"Command yielded code {result.returncode}",
                    "Stdout:",
                    result.stdout.decode(),
                    "Stderr:",
                    result.stderr.decode(),
                ]
            )
        )


@pytest.fixture(scope="session")
def grammar_raw_meds(
    tmp_path_factory: pytest.TempPathFactory,
    grammar_n_train: int,
    grammar_n_tuning: int,
    grammar_n_held_out: int,
) -> tuple[Path, Path]:
    """Build a raw-MEDS directory from the synthetic grammar + a task-labels directory.

    Returns ``(meds_root, task_labels_dir)``. The MEDS root has ``data/`` and ``metadata/``
    subdirs ready for ``MEICAR_process_data``, and ``task_labels_dir`` points at the per-task
    subfolder that contains ``{split}.parquet`` label files.
    """
    root = tmp_path_factory.mktemp("grammar_raw_meds")
    build_grammar_meds_dataset(
        root,
        n_train=grammar_n_train,
        n_tuning=grammar_n_tuning,
        n_held_out=grammar_n_held_out,
    )
    return root, root / "task_labels" / GRAMMAR_TASK_NAME


@pytest.fixture(scope="session")
def grammar_preprocessed(
    grammar_raw_meds: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Runs ``MEICAR_process_data`` on the grammar raw MEDS dir and returns the tensorized dir."""
    meds_dir, _ = grammar_raw_meds
    root = tmp_path_factory.mktemp("grammar_preprocessed")
    _run_and_check(
        [
            "MEICAR_process_data",
            f"input_dir={meds_dir!s}",
            f"intermediate_dir={root / 'intermediate'!s}",
            f"output_dir={root / 'output'!s}",
            "do_demo=True",
            "do_reshard=False",
        ]
    )
    return root / "output"


@pytest.fixture(scope="session")
def grammar_pretrained(
    grammar_preprocessed: Path,
    tmp_path_factory: pytest.TempPathFactory,
    grammar_max_seq_len: int,
    grammar_model_heads: int,
    grammar_model_head_dim: int,
    grammar_pretrain_epochs: int,
    grammar_pretrain_batch_size: int,
) -> Path:
    """Runs ``MEICAR_pretrain`` on the grammar preprocessed dataset.

    Uses a substantially bigger model than the default demo preset — see the ``--grammar-*``
    pytest options for defaults. Early stopping is effectively disabled (``patience=100000``)
    because the demo patience=3 kicks in after only a handful of validation checks against a
    noisy baseline.

    The budget is sized to dominate test cost on purpose: this fixture underpins the strict
    greedy-correctness assertion in ``test_grammar_cli_greedy_output_is_fully_grammar_valid``,
    which requires enough training depth to reach 100% held-out grammar validity under greedy
    decoding. Expected runtime: several minutes on a laptop CPU. Skip locally in favor of
    doctests during iteration.
    """
    output_dir = tmp_path_factory.mktemp("grammar_pretrained")
    _run_and_check(
        [
            "MEICAR_pretrain",
            "--config-name=_demo_pretrain",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.batch_size={grammar_pretrain_batch_size}",
            f"trainer.max_epochs={grammar_pretrain_epochs}",
            "trainer.overfit_batches=0",
            "trainer.callbacks.early_stopping.patience=100000",
            f"max_seq_len={grammar_max_seq_len}",
            f"lightning_module.model.gpt_kwargs.num_attention_heads={grammar_model_heads}",
            f"lightning_module.model.gpt_kwargs.attention_head_dim={grammar_model_head_dim}",
        ]
    )
    return output_dir


def _run_grammar_generate(
    output_dir: Path,
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    task_dir: Path,
    *,
    do_sample: bool,
    n_trajectories: int,
    batch_size: int,
    rolling_max_new_tokens: int,
) -> None:
    """Invoke ``MEICAR_generate_trajectories`` with the shared grammar fixture config.

    Centralized so the sampling and greedy fixtures only differ by ``do_sample`` and can't
    drift on other knobs.
    """
    _run_and_check(
        [
            "MEICAR_generate_trajectories",
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={grammar_pretrained!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.config.task_labels_dir={task_dir!s}",
            f"datamodule.batch_size={batch_size}",
            "trainer=demo",
            f"inference.N_trajectories_per_task_sample={n_trajectories}",
            f"inference.do_sample={str(do_sample).lower()}",
            f"rolling_generation.max_new_tokens={rolling_max_new_tokens}",
        ]
    )


@pytest.fixture(scope="session")
def grammar_generated_trajectories(
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
    grammar_n_trajectories: int,
    grammar_generate_batch_size: int,
    grammar_rolling_max_new_tokens: int,
) -> Path:
    """Runs ``MEICAR_generate_trajectories`` on the grammar-pretrained model with sampling.

    Uses a larger ``N_trajectories_per_task_sample`` than the demo default, with a
    ``batch_size`` deliberately not a multiple of N (stresses the #103 demux). Rolling
    generation is enabled with a budget above ``max_seq_len`` so the sliding-window path is
    exercised end-to-end.
    """
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("grammar_generated_trajectories")
    _run_grammar_generate(
        output_dir,
        grammar_pretrained,
        grammar_preprocessed,
        task_dir,
        do_sample=True,
        n_trajectories=grammar_n_trajectories,
        batch_size=grammar_generate_batch_size,
        rolling_max_new_tokens=grammar_rolling_max_new_tokens,
    )
    return output_dir


@pytest.fixture(scope="session")
def grammar_generated_trajectories_greedy(
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
    grammar_n_trajectories: int,
    grammar_generate_batch_size: int,
    grammar_rolling_max_new_tokens: int,
) -> Path:
    """Same as :func:`grammar_generated_trajectories` but with ``do_sample=False`` (greedy argmax).

    Greedy decoding gives fully deterministic outputs given the trained weights and prompt, so
    this fixture underpins the strict grammar-correctness assertion in
    ``test_grammar_cli_greedy_output_is_fully_grammar_valid``: if the model learned the grammar,
    its argmax continuation should be 100% grammar-valid.
    """
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("grammar_generated_trajectories_greedy")
    _run_grammar_generate(
        output_dir,
        grammar_pretrained,
        grammar_preprocessed,
        task_dir,
        do_sample=False,
        n_trajectories=grammar_n_trajectories,
        batch_size=grammar_generate_batch_size,
        rolling_max_new_tokens=grammar_rolling_max_new_tokens,
    )
    return output_dir

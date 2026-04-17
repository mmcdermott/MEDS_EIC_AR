"""Helpers for constructing a MEDS-format dataset from the synthetic grammar.

The pattern grammar lives in :mod:`tests.test_pattern_generation` as integer token sequences.
To drive the full ``MEICAR_process_data`` → ``MEICAR_pretrain`` → ``MEICAR_generate_trajectories``
CLI pipeline on it — issue #105 — we need to round-trip those integer sequences through MEDS's
``(subject_id, time, code, numeric_value)`` schema. This module owns that mapping.

Token ↔ MEDS-code mapping:

- ``SEP`` (integer 1) ↔ ``"GRAMMAR//SEP"``
- ``A[i]`` ↔ ``"GRAMMAR//A//{i}"`` for ``i in 0..3``
- ``B[i]`` ↔ ``"GRAMMAR//B//{i}"`` for ``i in 0..2``
- ``C[i]`` ↔ ``"GRAMMAR//C//{i}"`` for ``i in 0..8``

``TIMELINE//END`` / ``TIMELINE//START`` / ``TIMELINE//DELTA//*`` are **not** part of the grammar —
the CLI preprocessing pipeline adds those automatically based on timestamps, and the generation
pipeline can emit them because they're in the vocabulary. The grammar validation only cares about
the grammar-specific codes; non-grammar codes in the output are tolerated (skipped in the FSM
walk) so time deltas and the trained-in EOS don't cause false failures.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from tests.test_pattern_generation import (
    PROGRAMS,
    SEP,
    GrammarFSM,
    sample_sequence,
)

#: Mapping from grammar token integer → MEDS code string.
TOKEN_TO_CODE: dict[int, str] = {SEP: "GRAMMAR//SEP"}
for _name, _prog in PROGRAMS.items():
    for _i, _token in enumerate(_prog):
        TOKEN_TO_CODE[_token] = f"GRAMMAR//{_name}//{_i}"

#: Reverse mapping. Codes that aren't in this dict (``TIMELINE//END``, time deltas, etc.) are
#: simply not grammar tokens and get skipped by the FSM walker.
CODE_TO_TOKEN: dict[str, int] = {code: token for token, code in TOKEN_TO_CODE.items()}

#: All grammar codes in the order they'd first appear — used as the ``code_metadata`` rows so
#: every grammar token gets a vocabulary entry.
GRAMMAR_CODES: list[str] = list(TOKEN_TO_CODE.values())

# -------------------------------------------------------------------------
# Shared training-hyperparameter constants (issue #105).
#
# These are the knobs both the pretrain fixture (``conftest.py::grammar_pretrained``) and the
# generate fixture (``conftest.py::grammar_generated_trajectories``) need to agree on, and the
# assertions in ``tests/test_pattern_generation_cli.py`` also read them so we don't have two
# copies of "N=8" that silently drift apart. Module-level constants in a test helper are the
# least-awkward way to share these; a pytest fixture wrapping them would also work but adds
# indirection for no gain.
# -------------------------------------------------------------------------
#: ``max_seq_len`` override passed to ``MEICAR_pretrain``.
GRAMMAR_MAX_SEQ_LEN: int = 16
#: Number of attention heads. With ``GRAMMAR_MODEL_HEAD_DIM=32`` below:
#: ``hidden_size = heads * head_dim = 128``.
GRAMMAR_MODEL_HEADS: int = 4
#: Per-head dim. At small scale the model is capacity-limited on grammar generalization;
#: hidden_size=128 (plus more training data and more epochs, below) reliably produces an
#: unambiguous distributional held-out grammar signal that survives sampling variance.
GRAMMAR_MODEL_HEAD_DIM: int = 32
#: Pretrain epoch budget. Sized to dominate the overall fixture cost so the test's strict
#: assertions (greedy == 100% valid on held-out, sampling >= 50% passing rate at 50% validity)
#: have the training depth they need. In-process ``test_pattern_generation.py`` uses 400 steps
#: of batch=32 = 12,800 forward passes to get 100% greedy validity; we pay a similar budget here
#: (``n_train=256`` subjects x 400 epochs / batch=32 = 3200 steps x 32 = 102,400 forward passes)
#: through the Lightning + MEDS + Hydra stack. This test is intentionally the expensive
#: content-correctness regression test for the CLI pipeline; iterate locally via doctests and
#: let this run in pre-commit sweeps or CI.
GRAMMAR_PRETRAIN_EPOCHS: int = 400
#: Batch size for pretrain. Matches the in-process test's batch size.
GRAMMAR_PRETRAIN_BATCH_SIZE: int = 32
#: N trajectories generated per task-sample in the generate CLI call.
GRAMMAR_N_TRAJECTORIES: int = 8
#: Batch size for the generate CLI — deliberately not a multiple of ``GRAMMAR_N_TRAJECTORIES`` so
#: the demux from #103 is stressed in its cross-sample-group configuration.
GRAMMAR_GENERATE_BATCH_SIZE: int = 3
#: Rolling-generation budget for the generate CLI. Chosen strictly above the single-chunk cap
#: ``GRAMMAR_MAX_SEQ_LEN`` so at least one sample must cross a chunk boundary; see the rolling
#: content assertion in ``test_grammar_cli_rolling_actually_rolls_and_preserves_signal``.
GRAMMAR_ROLLING_MAX_NEW_TOKENS: int = 30


def build_grammar_meds_dataset(
    data_dir: Path,
    task_labels_dir: Path | None = None,
    *,
    n_train: int = 256,
    n_tuning: int = 16,
    n_held_out: int = 16,
    seed: int = 0,
    max_len: int = 24,
) -> None:
    """Write a raw-MEDS directory of grammar-generated subject timelines.

    Each subject's timeline is one grammar sequence sampled from
    :func:`tests.test_pattern_generation.sample_sequence`. Tokens become MEDS events with
    timestamps at fixed 1-hour spacing. Codes follow :data:`TOKEN_TO_CODE`. The output layout
    matches the MEDS spec (``meds.code_metadata_filepath`` is ``metadata/codes.parquet``): one
    parquet per split, a ``metadata/codes.parquet``, and a ``metadata/subject_splits.parquet``,
    so ``MEICAR_process_data`` can consume it without modification.

    Args:
        data_dir: Directory to write the MEDS files into. Created if it doesn't exist.
        n_train, n_tuning, n_held_out: Number of subjects per split.
        seed: RNG seed for per-subject grammar sampling. Subject ids are assigned
            deterministically from ``range(1, 10_000)`` and do not depend on ``seed``.
        max_len: Max token length passed to ``sample_sequence`` for each subject.
    """
    data_dir = Path(data_dir)
    (data_dir / "data" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "data" / "tuning").mkdir(parents=True, exist_ok=True)
    (data_dir / "data" / "held_out").mkdir(parents=True, exist_ok=True)
    (data_dir / "metadata").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
    subject_counter = iter(range(1, 10_000))

    split_counts = {"train": n_train, "tuning": n_tuning, "held_out": n_held_out}
    subject_splits_rows: list[dict] = []
    task_labels_by_split: dict[str, list[dict]] = {}

    for split, n_subjects in split_counts.items():
        rows: list[dict] = []
        split_task_rows: list[dict] = []
        for _ in range(n_subjects):
            subject_id = next(subject_counter)
            subject_splits_rows.append({"subject_id": subject_id, "split": split})
            tokens = sample_sequence(rng, max_len=max_len)
            for i, tok in enumerate(tokens):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": base_time + timedelta(hours=i),
                        "code": TOKEN_TO_CODE[tok],
                        "numeric_value": None,
                    }
                )
            # One task-labels row per subject — prediction_time is placed at the midpoint of
            # the timeline so the generator has at least one real event of history to condition
            # on and at least one real future event to compare against, regardless of how long
            # the sampled sequence ended up. With the shortest possible grammar sequence (a
            # single ``B`` program plus ``SEP`` = 4 tokens, last event at hour 3), a fixed
            # offset like 5 hours would fall past the last event. The label value is unused by
            # the generation path but the columns are required by MEDSTorchDataConfig.
            midpoint_hours = len(tokens) // 2
            split_task_rows.append(
                {
                    "subject_id": subject_id,
                    "prediction_time": base_time + timedelta(hours=midpoint_hours),
                    "boolean_value": False,
                    "integer_value": None,
                    "float_value": None,
                    "categorical_value": None,
                }
            )
        df = pl.DataFrame(
            rows,
            schema={
                "subject_id": pl.Int64,
                # tz-aware to match the Python ``datetime(..., tzinfo=UTC)`` used when building
                # ``rows``. A naive ``pl.Datetime("us")`` schema silently strips the timezone at
                # construction time, which "works" today but can flip to an error on polars
                # upgrades and produces parquet timestamps without the expected ``UTC`` metadata.
                "time": pl.Datetime("us", time_zone="UTC"),
                "code": pl.Utf8,
                "numeric_value": pl.Float32,
            },
        )
        df.write_parquet(data_dir / "data" / split / "0.parquet")
        task_labels_by_split[split] = split_task_rows

    # Code metadata: one row per grammar code. MEICAR_process_data also needs TIMELINE//END to be
    # addable by the preprocessing pipeline, but that lookup happens on the processed metadata
    # (output of MEICAR_process_data), not on the raw metadata we provide here — so we only need
    # to register the grammar codes.
    code_metadata = pl.DataFrame(
        {
            "code": GRAMMAR_CODES,
            "description": [f"Grammar token {c}" for c in GRAMMAR_CODES],
            "parent_codes": [[] for _ in GRAMMAR_CODES],
        },
        schema={
            "code": pl.Utf8,
            "description": pl.Utf8,
            "parent_codes": pl.List(pl.Utf8),
        },
    )
    code_metadata.write_parquet(data_dir / "metadata" / "codes.parquet")

    subject_splits = pl.DataFrame(
        subject_splits_rows,
        schema={"subject_id": pl.Int64, "split": pl.Utf8},
    )
    subject_splits.write_parquet(data_dir / "metadata" / "subject_splits.parquet")

    # Dataset metadata file — MEDSDataset requires this. Minimal content is fine.
    (data_dir / "metadata" / "dataset.json").write_text(
        '{"dataset_name": "grammar_integration_test", "dataset_version": "0.1.0", '
        '"etl_name": "tests/_grammar_meds.py", "etl_version": "0.1.0"}'
    )

    if task_labels_dir is not None:
        # Mirror meds_testing_helpers' task_labels layout: one parquet per task-split group.
        task_dir = Path(task_labels_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in task_labels_by_split.items():
            task_df = pl.DataFrame(
                rows,
                schema={
                    "subject_id": pl.Int64,
                    # Match the tz-aware convention used for the events ``time`` column above so
                    # downstream MEDS tooling sees consistent timestamp metadata on both files.
                    "prediction_time": pl.Datetime("us", time_zone="UTC"),
                    "boolean_value": pl.Boolean,
                    "integer_value": pl.Int64,
                    "float_value": pl.Float32,
                    "categorical_value": pl.Utf8,
                },
            )
            task_df.write_parquet(task_dir / f"{split}.parquet")


def grammar_tokens_from_output_df(df: pl.DataFrame) -> dict[int, list[int]]:
    """Extract per-subject grammar-token sequences from a generated-trajectory parquet.

    The output parquet has one row per emitted code, and the row-order in the parquet is taken
    as generation order. This is the contract ``MEICAR_generate_trajectories`` writes with
    today (rows are appended in the order tokens come out of the model), and the test suite
    relies on it — if a future change starts sorting or re-ordering rows on write, this helper
    will silently produce wrong sequences and the distinctness / grammar-signal assertions will
    break. A more durable alternative would be an explicit generation-order field; that's out
    of scope here (#105) but tracked as a follow-up.

    We filter to the grammar codes only (skipping ``TIMELINE//END``, time-delta tokens, and
    anything else the preprocessing added) and group by ``subject_id``.

    Args:
        df: A parquet read from ``<output_dir>/<split>/<trajectory_idx>.parquet``.

    Returns:
        Mapping from ``subject_id`` → list of grammar token ints in emission order.
    """
    codes_by_subject: dict[int, list[int]] = {}
    for row in df.iter_rows(named=True):
        code = row["code"]
        if code in CODE_TO_TOKEN:
            codes_by_subject.setdefault(row["subject_id"], []).append(CODE_TO_TOKEN[code])
    return codes_by_subject


def grammar_fsm_walk_is_valid(tokens: list[int]) -> tuple[bool, int]:
    """Run tokens through :class:`GrammarFSM` and return (is_valid, first_invalid_position).

    Returns ``(True, len(tokens))`` if every transition is grammar-valid, else
    ``(False, idx)`` where ``idx`` is the first invalid position.
    """
    fsm = GrammarFSM()
    first_invalid = fsm.walk(tokens)
    return first_invalid == len(tokens), first_invalid

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


def build_grammar_meds_dataset(
    data_dir: Path,
    task_labels_dir: Path | None = None,
    *,
    n_train: int = 24,
    n_tuning: int = 4,
    n_held_out: int = 4,
    seed: int = 0,
    max_len: int = 24,
    prediction_time_offset_hours: int = 5,
) -> None:
    """Write a raw-MEDS directory of grammar-generated subject timelines.

    Each subject's timeline is one grammar sequence sampled from
    :func:`tests.test_pattern_generation.sample_sequence`. Tokens become MEDS events with
    timestamps at fixed 1-hour spacing. Codes follow :data:`TOKEN_TO_CODE`. The output layout
    matches what :class:`meds_testing_helpers.dataset.MEDSDataset` writes (one parquet per
    split, a ``code_metadata.parquet``, and a ``subject_splits.parquet``), so
    ``MEICAR_process_data`` can consume it without modification.

    Args:
        data_dir: Directory to write the MEDS files into. Created if it doesn't exist.
        n_train, n_tuning, n_held_out: Number of subjects per split.
        seed: RNG seed for grammar sampling + subject-id assignment.
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
            # One task-labels row per subject — prediction_time sits inside the timeline so the
            # generator has real history to condition on and real future to generate. The label
            # value is unused by the generation path (generate_trajectories ignores it) but the
            # column is required by MEDSTorchDataConfig.
            split_task_rows.append(
                {
                    "subject_id": subject_id,
                    "prediction_time": base_time + timedelta(hours=prediction_time_offset_hours),
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
                "time": pl.Datetime("us"),
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
                    "prediction_time": pl.Datetime("us"),
                    "boolean_value": pl.Boolean,
                    "integer_value": pl.Int64,
                    "float_value": pl.Float32,
                    "categorical_value": pl.Utf8,
                },
            )
            task_df.write_parquet(task_dir / f"{split}.parquet")


def grammar_tokens_from_output_df(df: pl.DataFrame) -> dict[int, list[int]]:
    """Extract per-subject grammar-token sequences from a generated-trajectory parquet.

    The output parquet has one row per emitted code, in emission order. We filter to the
    grammar codes only (skipping ``TIMELINE//END``, time-delta tokens, and anything else the
    preprocessing added) and group by ``subject_id``.

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

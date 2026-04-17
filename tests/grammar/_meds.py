"""Round-trip the synthetic grammar through MEDS format for end-to-end CLI testing.

The grammar itself lives in :mod:`tests.grammar._grammar` as integer token sequences. To drive
the full ``MEICAR_process_data`` → ``MEICAR_pretrain`` → ``MEICAR_generate_trajectories`` CLI
pipeline on it — issue #105 — we need to serialize those sequences as MEDS-format subject
timelines. This module owns that serialization.

Token ↔ MEDS-code mapping:

- ``SEP`` (integer 1) ↔ ``"GRAMMAR//SEP"``
- ``A[i]`` ↔ ``"GRAMMAR//A//{i}"`` for ``i in 0..3``
- ``B[i]`` ↔ ``"GRAMMAR//B//{i}"`` for ``i in 0..2``
- ``C[i]`` ↔ ``"GRAMMAR//C//{i}"`` for ``i in 0..8``

``TIMELINE//END`` / ``TIMELINE//START`` / ``TIMELINE//DELTA//*`` are **not** part of the grammar
— the CLI preprocessing pipeline adds those automatically based on timestamps, and the
generation pipeline can emit them because they're in the vocabulary. The grammar validation only
cares about the grammar-specific codes; non-grammar codes in the output are tolerated (skipped
in the FSM walk) so time deltas and the trained-in EOS don't cause false failures.

Writing the MEDS layout is delegated to :class:`meds_testing_helpers.dataset.MEDSDataset`:
construct in memory, call ``.write(out_dir)``. That removes a lot of the manual parquet /
``dataset.json`` / subject-splits code we'd otherwise write by hand, and keeps the schema tied
to the canonical :mod:`meds` schemas (``DataSchema``, ``LabelSchema``, ``CodeMetadataSchema``,
``SubjectSplitSchema``) so we don't drift out of spec on a future MEDS bump.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from meds import (
    DatasetMetadataSchema,
    data_subdirectory,
    held_out_split,
    train_split,
    tuning_split,
)
from meds_testing_helpers.dataset import MEDSDataset

from tests.grammar._grammar import PROGRAMS, SEP, sample_sequence

#: Single task name used for the generated trajectories task-label file. The CLI's
#: ``datamodule.config.task_labels_dir`` points at the subfolder
#: ``task_labels/<GRAMMAR_TASK_NAME>/`` where :class:`MEDSDataset.write` lays down the per-shard
#: label parquets.
GRAMMAR_TASK_NAME: str = "grammar_continue_from_sep"

#: Mapping from grammar token integer → MEDS code string.
TOKEN_TO_CODE: dict[int, str] = {SEP: "GRAMMAR//SEP"}
for _name, _prog in PROGRAMS.items():
    for _i, _token in enumerate(_prog):
        TOKEN_TO_CODE[_token] = f"GRAMMAR//{_name}//{_i}"

#: Reverse mapping. Codes that aren't in this dict (``TIMELINE//END``, time deltas, etc.) are
#: simply not grammar tokens and get skipped by the FSM walker.
CODE_TO_TOKEN: dict[str, int] = {code: token for token, code in TOKEN_TO_CODE.items()}

#: All grammar codes, used as ``code_metadata`` rows so every grammar token has a vocab entry.
GRAMMAR_CODES: list[str] = list(TOKEN_TO_CODE.values())


def _sample_multi_program_sequence(rng: random.Random, max_len: int) -> list[int]:
    """Sample a grammar sequence that contains at least two SEP tokens.

    We need at least two programs per subject so ``prediction_time`` can be placed at the first
    program boundary with real "future" content left to generate. Single-program sequences (e.g.
    ``[B[0], B[1], B[2], SEP]``) are the degenerate case where no such boundary exists.
    """
    tokens = sample_sequence(rng, max_len=max_len)
    while tokens.count(SEP) < 2:
        tokens = sample_sequence(rng, max_len=max_len)
    return tokens


def build_grammar_meds_dataset(
    data_dir: Path,
    *,
    n_train: int,
    n_tuning: int,
    n_held_out: int,
    seed: int = 0,
    max_len: int = 24,
) -> MEDSDataset:
    """Build and write a MEDS-format synthetic grammar dataset.

    Each subject's timeline is one grammar sequence sampled from
    :func:`tests.grammar._grammar.sample_sequence` and constrained to contain at least two
    programs so there's a clean program-boundary to anchor the task-label ``prediction_time``
    at. Tokens become MEDS events with timestamps at fixed 1-hour spacing. Codes follow
    :data:`TOKEN_TO_CODE`.

    Writes under ``data_dir``:

    - ``data/{split}/0.parquet`` — one shard per split in the standard MEDS layout.
    - ``metadata/codes.parquet`` — one row per grammar code.
    - ``metadata/subject_splits.parquet`` — subject → split mapping.
    - ``metadata/dataset.json`` — minimal ``DatasetMetadataSchema`` identifying this as a
      grammar fixture.
    - ``task_labels/{GRAMMAR_TASK_NAME}/{split}.parquet`` — a single task label per subject
      with ``prediction_time`` placed strictly between the first ``SEP`` event and the next
      grammar event (see the ``prediction_time`` comment below for the off-by-one rationale).

    Args:
        data_dir: Directory to write the MEDS files into. Created if it doesn't exist.
        n_train, n_tuning, n_held_out: Number of subjects per split.
        seed: RNG seed for per-subject grammar sampling. Subject ids are assigned
            deterministically from ``range(1, 10_000)`` and do not depend on ``seed``.
        max_len: Max token length passed to ``sample_sequence`` for each subject.

    Returns:
        The :class:`MEDSDataset` that was just written, for callers that want to inspect it
        without re-reading from disk.
    """
    rng = random.Random(seed)
    base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
    subject_counter = iter(range(1, 10_000))

    split_counts = {
        train_split: n_train,
        tuning_split: n_tuning,
        held_out_split: n_held_out,
    }
    subject_splits_rows: list[dict] = []
    event_rows_by_split: dict[str, list[dict]] = {split: [] for split in split_counts}
    task_label_rows_by_split: dict[str, list[dict]] = {split: [] for split in split_counts}

    for split, n_subjects in split_counts.items():
        for _ in range(n_subjects):
            subject_id = next(subject_counter)
            subject_splits_rows.append({"subject_id": subject_id, "split": split})

            tokens = _sample_multi_program_sequence(rng, max_len=max_len)
            for i, tok in enumerate(tokens):
                event_rows_by_split[split].append(
                    {
                        "subject_id": subject_id,
                        "time": base_time + timedelta(hours=i),
                        "code": TOKEN_TO_CODE[tok],
                        "numeric_value": None,
                    }
                )

            # Place ``prediction_time`` strictly **between** the first SEP event and the next
            # grammar event. Why not exactly on the hour of the next event? meds-torchdata's
            # ``MEDSPytorchDataset._prepare_task_label_schema`` uses
            # ``pl.col("time").search_sorted(prediction_time, side="right")`` to compute the
            # end-of-prompt index, which **includes** events at ``time == prediction_time`` in
            # the prompt. An integer-hour ``prediction_time`` sitting exactly on an event would
            # silently pull that event into the prompt and shift the "first generated token"
            # forward by one event. Placing it at ``sep_hour + 30min`` avoids the ambiguity:
            # no event sits at that time, ``side`` doesn't change the result. See issue #111
            # for the full investigation.
            first_sep_idx = tokens.index(SEP)
            task_label_rows_by_split[split].append(
                {
                    "subject_id": subject_id,
                    "prediction_time": base_time + timedelta(hours=first_sep_idx, minutes=30),
                    "boolean_value": False,
                    "integer_value": None,
                    "float_value": None,
                    "categorical_value": None,
                }
            )

    # Build the MEDSDataset in memory, then write it. MEDSDataset's shard-name-to-filename
    # mapping is "shard_name → data/<shard_name>.parquet", so shard names of the form
    # "<split>/0" produce the "data/<split>/0.parquet" layout the CLI expects.
    data_shards = {
        f"{split}/0": pl.DataFrame(
            rows,
            schema={
                "subject_id": pl.Int64,
                # tz-aware to match the Python ``datetime(..., tzinfo=UTC)`` used when building
                # ``rows``. A naive ``pl.Datetime("us")`` schema silently strips the timezone
                # at construction time, which "works" today but can flip to an error on polars
                # upgrades and produces parquet timestamps without the expected UTC metadata.
                "time": pl.Datetime("us", time_zone="UTC"),
                "code": pl.Utf8,
                "numeric_value": pl.Float32,
            },
        )
        for split, rows in event_rows_by_split.items()
    }

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

    subject_splits = pl.DataFrame(
        subject_splits_rows,
        schema={"subject_id": pl.Int64, "split": pl.Utf8},
    )

    # One task with per-split shard names. MEDSDataset writes each at
    # ``task_labels/<task_name>/<shard_name>.parquet``. ``PL_LABEL_SCHEMA`` is the canonical
    # polars schema for MEDS task labels so we don't redefine it here.
    task_labels = {
        GRAMMAR_TASK_NAME: {
            split: pl.DataFrame(rows, schema=MEDSDataset.PL_LABEL_SCHEMA).with_columns(
                # Upgrade ``prediction_time`` from the canonical naive schema to tz-aware to
                # match the event ``time`` column written above.
                pl.col("prediction_time").dt.replace_time_zone("UTC"),
            )
            for split, rows in task_label_rows_by_split.items()
        }
    }

    dataset_metadata = DatasetMetadataSchema(
        dataset_name="grammar_integration_test",
        dataset_version="0.1.0",
        etl_name="tests/grammar/_meds.py",
        etl_version="0.1.0",
    )

    dataset = MEDSDataset(
        data_shards=data_shards,
        dataset_metadata=dataset_metadata,
        code_metadata=code_metadata,
        subject_splits=subject_splits,
        task_labels=task_labels,
    )
    dataset.write(Path(data_dir))
    return dataset


def grammar_tokens_from_output_df(df: pl.DataFrame) -> dict[int, list[int]]:
    """Extract per-subject grammar-token sequences from a generated-trajectory parquet.

    The output parquet has one row per emitted code, and the row-order in the parquet is taken
    as generation order. This is the contract ``MEICAR_generate_trajectories`` writes with
    today (rows are appended in the order tokens come out of the model); if a future change
    starts sorting or re-ordering rows on write, this helper will silently produce wrong
    sequences and the distinctness / grammar-signal assertions will break. An explicit
    generation-order field would be more durable; tracked as a follow-up.

    We filter to the grammar codes only (skipping ``TIMELINE//END``, time-delta tokens, and
    anything else the preprocessing added) and group by ``subject_id``.
    """
    codes_by_subject: dict[int, list[int]] = {}
    for row in df.iter_rows(named=True):
        code = row["code"]
        if code in CODE_TO_TOKEN:
            codes_by_subject.setdefault(row["subject_id"], []).append(CODE_TO_TOKEN[code])
    return codes_by_subject


def prompt_grammar_tokens_by_subject(
    raw_meds_dir: Path, task_labels_dir: Path, split: str
) -> dict[int, list[int]]:
    """Return the grammar tokens of each subject's prompt in ``split``.

    "Prompt" here = the events strictly before ``prediction_time``. These are the tokens the
    model conditions on before emitting anything. Walking them through :class:`GrammarFSM`
    gives the FSM state at generation time, which is the starting state to evaluate the
    generated continuation against — otherwise a fresh FSM at ``BETWEEN`` would reject any
    in-program continuation (e.g. ``A[2]``) as invalid even though it's the grammar-correct
    continuation from wherever the prompt left off.

    Args:
        raw_meds_dir: The MEDS root that :func:`build_grammar_meds_dataset` wrote (has
            ``data/{split}/0.parquet`` and ``metadata/``).
        task_labels_dir: The directory containing ``{split}.parquet`` task-label files for
            this task — typically ``<meds_root>/task_labels/<GRAMMAR_TASK_NAME>/``.
        split: One of :data:`meds.train_split`, :data:`meds.tuning_split`,
            :data:`meds.held_out_split`.

    Returns:
        ``{subject_id: [grammar_token_int, ...]}`` for every subject in the split, in
        chronological (event-time) order. Aux codes aren't in the raw data we wrote, so the
        filter-by-``CODE_TO_TOKEN`` is defensive, not load-bearing.
    """
    events_df = pl.read_parquet(raw_meds_dir / data_subdirectory / split / "0.parquet")
    task_df = pl.read_parquet(Path(task_labels_dir) / f"{split}.parquet")

    prompt_by_subject: dict[int, list[int]] = {}
    for row in task_df.iter_rows(named=True):
        subject_id = row["subject_id"]
        prediction_time = row["prediction_time"]
        prompt_df = events_df.filter(
            (pl.col("subject_id") == subject_id) & (pl.col("time") < prediction_time)
        ).sort("time")
        prompt_by_subject[subject_id] = [
            CODE_TO_TOKEN[code] for code in prompt_df["code"].to_list() if code in CODE_TO_TOKEN
        ]
    return prompt_by_subject

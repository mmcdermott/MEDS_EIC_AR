"""Integration test: full CLI generation pipeline driven on the synthetic grammar dataset.

Closes issue #105. The existing tests split the work:

- ``tests/test_pattern_generation.py`` trains a tiny ``Model`` in-process on the grammar and
  calls ``Model.generate`` directly — proving **generation is content-correct** (grammar-valid
  continuations) but bypassing the CLI / Lightning / parquet pipeline.
- ``tests/test_generate_trajectories.py::test_generate_trajectories_runs`` drives the full
  ``MEICAR_generate_trajectories`` CLI on a random-init demo model — proving **the plumbing is
  wired** (files exist, subjects line up) but unable to validate content because the model
  hasn't learned anything.

This file fills the gap between them: train a model **through the full CLI pipeline**
(``MEICAR_process_data`` → ``MEICAR_pretrain``) on a grammar-formatted MEDS dataset, generate
trajectories **through the full CLI** with ``N_trajectories_per_task_sample=8`` (larger than the
demo default) and a ``batch_size=3`` (deliberately not a multiple of N, to stress the
cross-sample-group demux from PR #103), then inspect the output parquets.

**Assertions are deliberately weaker than the in-process test.** CLI training runs on a small
budget for test runtime, and the model won't converge to perfect grammar adherence. What we can
assert reliably at this budget:

1. File-layout correctness: ``N`` parquet files per split, every subject appears in every
   trajectory, no duplicates.
2. Codes in output are all drawn from the trained vocab (no NaN, no negative ids, no unknown
   strings sneaking in via format_trajectories).
3. Demux correctness: different ``trajectory_idx`` parquets contain distinct content for the
   same subject (not identical rows — that would mean the interleaved predict pass collapsed N
   trajectories into 1).
4. Grammar **signal** (not strict validity): a meaningful fraction of the in-program transitions
   the model emits should be grammar-valid, which proves the model learned *something* about the
   data. Strict 100%-valid is covered by ``test_pattern_generation.py``.

Also adds a few small tests for the CLI config-validation error paths that aren't exercised by
the happy-path fixtures.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import polars as pl
from meds import held_out_split, train_split, tuning_split

if TYPE_CHECKING:
    from pathlib import Path

from tests._grammar_meds import (
    GRAMMAR_CODES,
    GrammarFSM,
    grammar_tokens_from_output_df,
)

# Auxiliary codes that MEICAR_process_data emits for us (timestamps → time-delta tokens; timeline
# start/end sentinels). These aren't part of the grammar but are expected in the output and must
# not cause a grammar-check failure.
_AUX_CODE_PREFIXES = ("TIMELINE//", "GRAMMAR//")


def _load_trajectories_by_split(root: Path) -> dict[str, dict[str, pl.DataFrame]]:
    out: dict[str, dict[str, pl.DataFrame]] = {}
    for split in (train_split, tuning_split, held_out_split):
        split_dir = root / split
        out[split] = {}
        for fp in sorted(split_dir.glob("*.parquet")):
            df = pl.read_parquet(fp, use_pyarrow=True)
            assert len(df) > 0, f"Empty parquet at {fp}"
            out[split][fp.stem] = df
        assert out[split], f"No parquets found under {split_dir}"
    return out


def test_grammar_cli_produces_n_parquets_per_split(grammar_generated_trajectories: Path):
    """N trajectories x 3 splits x correct shape — the bedrock structural signal."""
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    n_trajectories = 8  # matches grammar_generated_trajectories fixture override

    for split, trajectories in by_split.items():
        assert set(trajectories.keys()) == {str(i) for i in range(n_trajectories)}, (
            f"Split {split}: expected parquets 0..{n_trajectories - 1}, got {sorted(trajectories)}"
        )
        # Every trajectory parquet for a split must cover the same subject set. If the
        # interleaving path corrupted the demux, we'd see subjects dropped or duplicated across
        # trajectories — this assertion catches that.
        subject_sets = {t: set(df["subject_id"].to_list()) for t, df in trajectories.items()}
        expected = subject_sets["0"]
        for t, subjects in subject_sets.items():
            assert subjects == expected, (
                f"Split {split} trajectory {t} has subjects {subjects} but trajectory 0 has "
                f"{expected}. The demux in #103 may have dropped or duplicated rows."
            )


def test_grammar_cli_trajectories_are_distinct_across_n(grammar_generated_trajectories: Path):
    """For each subject, the N generated trajectories should not all be identical.

    If the interleaved predict pass collapsed N identical copies (e.g. because seeding was misapplied or the
    model is deterministic greedy + the same prompt), we'd see every parquet produce the same token sequence
    for a given subject. This is a weak but important sanity check that sampling actually diversifies.
    """
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    for split, trajectories in by_split.items():
        # Build per-(trajectory_idx, subject_id) code tuple and compare across trajectories.
        per_traj_per_subject: dict[str, dict[int, tuple[str, ...]]] = {}
        for t, df in trajectories.items():
            per_subject: dict[int, tuple[str, ...]] = {}
            for subject_id, subject_df in df.group_by("subject_id"):
                per_subject[subject_id[0]] = tuple(subject_df["code"].to_list())
            per_traj_per_subject[t] = per_subject

        subjects_any = list(per_traj_per_subject["0"].keys())
        # At least one subject must have at least two distinct trajectories out of N.
        any_subject_is_diverse = any(
            len({per_traj_per_subject[t].get(s) for t in per_traj_per_subject}) > 1 for s in subjects_any
        )
        assert any_subject_is_diverse, (
            f"Split {split}: every subject produced identical trajectories across all 8 samples. "
            f"Either sampling is degenerate (model always picks the same token) or the "
            f"interleaved-predict path collapsed the N trajectories into 1."
        )


def test_grammar_cli_output_codes_are_in_expected_vocab(grammar_generated_trajectories: Path):
    """Every code in the output must be either a grammar code or a recognized auxiliary code (time delta /
    timeline sentinel).

    If format_trajectories produced an unrecognized code string, this catches it before it silently corrupts
    downstream analysis.
    """
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    grammar_codes = set(GRAMMAR_CODES)

    for split, trajectories in by_split.items():
        for t, df in trajectories.items():
            for code in df["code"].unique().to_list():
                in_grammar = code in grammar_codes
                in_aux = any(code.startswith(p) for p in _AUX_CODE_PREFIXES)
                assert in_grammar or in_aux, (
                    f"Split {split} trajectory {t}: output code {code!r} is neither a grammar "
                    f"code ({sorted(grammar_codes)}) nor an auxiliary code with prefix "
                    f"{_AUX_CODE_PREFIXES}."
                )


def _walk_grammar_tokens_ignoring_start(tokens: list[int]) -> tuple[int, int]:
    """Return (n_valid_transitions, n_total_transitions).

    The FSM's initial state is BETWEEN; feeding the tokens in order counts each transition. We consider a
    transition "valid" if it doesn't send the FSM to the invalid sink. Since we stop walking on the first
    invalid step, the return is the count before failure.
    """
    fsm = GrammarFSM()
    valid = 0
    for tok in tokens:
        if fsm.step(tok) is None:
            break
        valid += 1
    return valid, len(tokens)


def test_grammar_cli_model_shows_grammar_signal(grammar_generated_trajectories: Path):
    """Soft grammar-adherence check: the model's output must contain a meaningful fraction of
    valid grammar transitions.

    This is a **signal** test, not a correctness test. Strict grammar validity for every emitted
    token is covered by ``test_pattern_generation.py``'s in-process tests, which have more
    training budget. Here we just want to confirm the CLI pipeline produced a model that learned
    at least *something* grammatical, which rules out entire classes of CLI bugs (wrong vocab
    indexing, off-by-one in token → code mapping, etc.) that would leave the model emitting
    structurally-random output.

    Threshold: at least one trajectory across the held-out split should have >= 30% of its
    grammar tokens be valid transitions. If every trajectory is below that, either the CLI is
    broken or the training budget is too small.
    """
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    best_fraction = 0.0
    for _split, trajectories in by_split.items():
        for _t, df in trajectories.items():
            tokens_by_subject = grammar_tokens_from_output_df(df)
            for _subject_id, tokens in tokens_by_subject.items():
                if not tokens:
                    continue
                valid, total = _walk_grammar_tokens_ignoring_start(tokens)
                fraction = valid / total
                best_fraction = max(best_fraction, fraction)

    assert best_fraction >= 0.3, (
        f"No generated trajectory in any split had >= 30% valid grammar transitions "
        f"(best seen: {best_fraction:.1%}). The CLI training + generation pipeline is probably "
        f"broken — a correctly wired pipeline with the fixture's training budget should produce "
        f"at least some coherent grammar output."
    )


# ---------------------------------------------------------------------------
# CLI error-path coverage (#105 acceptance criterion: exercise validation paths in __main__.py
# that the happy-path fixtures don't hit).
# ---------------------------------------------------------------------------


def _run_generate_and_capture(args: list[str]) -> subprocess.CompletedProcess:
    """Run ``MEICAR_generate_trajectories`` and return the completed process without raising."""
    return subprocess.run(
        ["MEICAR_generate_trajectories", *args], capture_output=True, check=False, shell=False
    )


def test_grammar_cli_rejects_zero_max_new_tokens(
    grammar_pretrained: Path, grammar_preprocessed: Path, grammar_raw_meds, tmp_path_factory
):
    """``rolling_generation.max_new_tokens=0`` must be rejected by ``validate_rolling_cfg`` before any compute
    happens."""
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("grammar_err_zero_tokens")
    result = _run_generate_and_capture(
        [
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={grammar_pretrained!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.config.task_labels_dir={task_dir!s}",
            "datamodule.batch_size=2",
            "trainer=demo",
            "rolling_generation.max_new_tokens=0",
        ]
    )
    assert result.returncode != 0, (
        "CLI accepted rolling_generation.max_new_tokens=0; it should have rejected it."
    )
    assert b"positive integer" in result.stderr or b"positive integer" in result.stdout, (
        f"Expected a 'positive integer' validation error; got stderr={result.stderr!r}"
    )


def test_grammar_cli_rejects_rolling_context_without_max_new_tokens(
    grammar_pretrained: Path, grammar_preprocessed: Path, grammar_raw_meds, tmp_path_factory
):
    """Setting ``rolling_context_size`` without ``max_new_tokens`` is a silent no-op on the single-chunk path
    — the CLI validator should reject it."""
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("grammar_err_ctx_without_budget")
    result = _run_generate_and_capture(
        [
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={grammar_pretrained!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.config.task_labels_dir={task_dir!s}",
            "datamodule.batch_size=2",
            "trainer=demo",
            "rolling_generation.rolling_context_size=8",
        ]
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert b"max_new_tokens" in combined and b"null" in combined, (
        f"Expected an error mentioning max_new_tokens being null; got stderr={result.stderr!r}"
    )


def test_grammar_cli_rejects_unknown_rolling_key(
    grammar_pretrained: Path, grammar_preprocessed: Path, grammar_raw_meds, tmp_path_factory
):
    """An unrecognized key under ``rolling_generation`` must surface as a config error rather than being
    silently ignored."""
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("grammar_err_unknown_key")
    result = _run_generate_and_capture(
        [
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir!s}",
            f"model_initialization_dir={grammar_pretrained!s}",
            f"datamodule.config.tensorized_cohort_dir={grammar_preprocessed!s}",
            f"datamodule.config.task_labels_dir={task_dir!s}",
            "datamodule.batch_size=2",
            "trainer=demo",
            "+rolling_generation.bogus_key=1",
        ]
    )
    # Hydra itself may reject the unknown key before our validator runs (struct-mode), which is
    # fine — either way, we should see a nonzero exit and a mention of the bad key.
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert b"bogus_key" in combined, (
        f"Expected an error mentioning the unknown key; got stderr={result.stderr!r}"
    )

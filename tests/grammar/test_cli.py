"""Integration test: full CLI generation pipeline on the synthetic grammar (closes issue #105).

Sits between the existing narrower tests:

- :mod:`tests.grammar.test_in_process` trains a tiny ``Model`` on the grammar directly and calls
  ``Model.generate`` — proving **generation is content-correct** but bypassing the CLI pipeline.
- ``tests/test_generate_trajectories.py::test_generate_trajectories_runs`` drives the full
  ``MEICAR_generate_trajectories`` CLI on a random-init demo model — proving **the plumbing is
  wired** but unable to validate content because the model hasn't learned anything.

This file fills the gap: train a model **through the full CLI pipeline**
(``MEICAR_process_data`` → ``MEICAR_pretrain``) on a grammar-formatted MEDS dataset, generate
trajectories **through the full CLI**, then inspect the output parquets. Fixture knobs
(trajectories per sample, batch size, rolling budget, ...) are pytest CLI options with
``--grammar-*`` prefixes — see ``tests/grammar/conftest.py`` for the full list. All constants
previously hardcoded here are now fixtures.

Assertions span plumbing and content:

1. File-layout correctness: ``N`` parquet files per split, every subject appears in every
   trajectory, no duplicates.
2. Codes in output are all drawn from the trained vocab (no NaN, no negative ids, no unknown
   strings sneaking in via format_trajectories).
3. Demux correctness: different ``trajectory_idx`` parquets contain distinct content for the
   same subject (not identical rows — that would mean the interleaved predict pass collapsed N
   trajectories into 1).
4. **Sampling grammar signal** (distributional): at least half of held-out (subject, trajectory)
   samples clear a per-sample 50% grammar-validity threshold on the ``do_sample=True`` fixture.
5. **Greedy strict validity**: every generated grammar token in every held-out sample under
   ``do_sample=False`` must be a valid FSM continuation from the prompt's end-state. Strict 100%
   validity, no averaging, no tolerance — byte-for-byte equivalent in strictness to the
   in-process :mod:`tests.grammar.test_in_process` tests' ``GrammarFSM().walk(seq) == len(seq)``.
6. **Rolling content**: at least one sample exceeded the single-chunk cap (proving rolling was
   exercised) and signal survives rolling at the same threshold as (4).

Also adds three CLI config-validation error-path tests for ``__main__.py``'s ``validate_rolling_cfg``.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import polars as pl
from meds import held_out_split, train_split, tuning_split

from tests.grammar._grammar import GrammarFSM
from tests.grammar._meds import (
    CODE_TO_TOKEN,
    GRAMMAR_CODES,
    grammar_tokens_from_output_df,
    prompt_grammar_tokens_by_subject,
)

if TYPE_CHECKING:
    from pathlib import Path

# Auxiliary codes that MEICAR_process_data emits for us (timestamps → time-delta tokens; timeline
# start/end sentinels). These aren't part of the grammar but are expected in the output and must
# not cause a grammar-check failure. ``GRAMMAR//`` is **not** listed here: grammar codes must pass
# the exact-membership check against ``GRAMMAR_CODES`` so a stray ``GRAMMAR//bogus`` code can't
# slip through and mask a token→code mapping bug.
_AUX_CODE_PREFIXES = ("TIMELINE//",)

#: Per-sample threshold: a sample is "passing" if this fraction of its grammar-token transitions
#: are FSM-valid, with the FSM pre-advanced by the subject's prompt tokens. 0.50 is comfortably
#: above chance (random picks from a ~50-entry vocab give ~6% per-step valid rate) and inside
#: what the CLI budget hits routinely; the in-process test reaches ~0.9 greedy with more training.
_PER_SAMPLE_VALIDITY_THRESHOLD: float = 0.50

#: Minimum fraction of held-out (subject, trajectory) samples that must clear the per-sample
#: threshold. Using a passing-rate distributional metric (instead of "best seen") rules out the
#: "one lucky sample masks universal garbage" failure mode.
_MIN_FRACTION_SAMPLES_PASSING: float = 0.50


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


def _fsm_advanced_through_prompt(prompt_tokens: list[int]) -> GrammarFSM:
    """Build a :class:`GrammarFSM` and walk it through the subject's prompt tokens.

    The returned FSM's state is the state the model was conditioning on when it began generating. Scoring the
    generated continuation from this state lets us judge strict validity of every single generated token. If
    the prompt itself walks invalid, something is wrong with the data setup — raise rather than silently mask.
    """
    fsm = GrammarFSM()
    for i, tok in enumerate(prompt_tokens):
        if fsm.step(tok) is None:
            raise AssertionError(
                f"Prompt tokens are not grammar-valid — data setup bug. First invalid at "
                f"position {i}; prompt={prompt_tokens}"
            )
    return fsm


def _walk_generated_strictly(prompt_tokens: list[int], generated_tokens: list[int]) -> tuple[int, int]:
    """Walk ``generated_tokens`` through an FSM pre-advanced by ``prompt_tokens``."""
    fsm = _fsm_advanced_through_prompt(prompt_tokens)
    valid = 0
    for tok in generated_tokens:
        if fsm.step(tok) is None:
            break
        valid += 1
    return valid, len(generated_tokens)


def test_grammar_cli_produces_n_parquets_per_split(
    grammar_generated_trajectories: Path, grammar_n_trajectories: int
):
    """N trajectories x 3 splits x correct shape — the bedrock structural signal."""
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    for split, trajectories in by_split.items():
        assert set(trajectories.keys()) == {str(i) for i in range(grammar_n_trajectories)}, (
            f"Split {split}: expected parquets 0..{grammar_n_trajectories - 1}, got {sorted(trajectories)}"
        )
        subject_sets = {t: set(df["subject_id"].to_list()) for t, df in trajectories.items()}
        expected = subject_sets["0"]
        for t, subjects in subject_sets.items():
            assert subjects == expected, (
                f"Split {split} trajectory {t} has subjects {subjects} but trajectory 0 has "
                f"{expected}. The demux in #103 may have dropped or duplicated rows."
            )


def test_grammar_cli_trajectories_are_distinct_across_n(
    grammar_generated_trajectories: Path, grammar_n_trajectories: int
):
    """For each subject, the N generated trajectories should not all be identical.

    If the interleaved predict pass collapsed N identical copies (e.g. because seeding was misapplied or the
    model is deterministic greedy + the same prompt), we'd see every parquet produce the same token sequence
    for a given subject.
    """
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    for split, trajectories in by_split.items():
        # Sort by ``time`` within each subject so the extracted sequence is a stable function of
        # generation order — polars' ``group_by`` doesn't guarantee intra-group row order without
        # ``maintain_order=True``, and if a reorder swapped two codes we'd spuriously detect
        # "diversity" even for identical runs.
        per_traj_per_subject: dict[str, dict[int, tuple[str, ...]]] = {}
        for t, df in trajectories.items():
            per_subject: dict[int, tuple[str, ...]] = {}
            sorted_df = df.sort(["subject_id", "time"])
            for subject_id, subject_df in sorted_df.group_by("subject_id", maintain_order=True):
                per_subject[subject_id[0]] = tuple(subject_df["code"].to_list())
            per_traj_per_subject[t] = per_subject

        subjects_any = set(per_traj_per_subject["0"].keys())
        for t, per_subject in per_traj_per_subject.items():
            assert set(per_subject.keys()) == subjects_any, (
                f"Split {split} trajectory {t} has subjects {set(per_subject.keys())} but "
                f"trajectory 0 has {subjects_any} — the demux dropped or added subjects."
            )

        any_subject_is_diverse = any(
            len({per_traj_per_subject[t][s] for t in per_traj_per_subject}) > 1 for s in subjects_any
        )
        assert any_subject_is_diverse, (
            f"Split {split}: every subject produced identical trajectories across all "
            f"{grammar_n_trajectories} samples. Either sampling is degenerate or the "
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


def test_grammar_cli_model_shows_grammar_signal(
    grammar_generated_trajectories: Path, grammar_raw_meds: tuple[Path, Path]
):
    """Distributional grammar-adherence check on the held-out split (sampling fixture).

    Per-sample metric: (# of valid FSM transitions) / (# of generated grammar tokens), with the
    FSM pre-advanced through the subject's prompt tokens so the starting state is exactly what
    the model conditioned on at generation time.

    Assertion: at least ``_MIN_FRACTION_SAMPLES_PASSING`` of held-out (subject, trajectory)
    pairs clear ``_PER_SAMPLE_VALIDITY_THRESHOLD``. Rules out the "one lucky sample masks
    universal garbage" failure mode without overreacting to sampling noise on individual rows.
    """
    raw_meds_dir, task_labels_dir = grammar_raw_meds
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    prompts_by_subject = prompt_grammar_tokens_by_subject(raw_meds_dir, task_labels_dir, held_out_split)

    # Restrict to held-out: train-split trajectories could exhibit apparent grammar signal from
    # memorization, which wouldn't prove the model generalized.
    fractions: list[float] = []
    for _t, df in by_split[held_out_split].items():
        tokens_by_subject = grammar_tokens_from_output_df(df)
        for subject_id, tokens in tokens_by_subject.items():
            if not tokens:
                continue
            valid, total = _walk_generated_strictly(prompts_by_subject[subject_id], tokens)
            if total == 0:
                continue
            fractions.append(valid / total)

    assert fractions, (
        "No held-out samples produced any grammar tokens. The fixture generates nothing or "
        "generates only non-grammar codes — both indicate a broken pipeline."
    )
    passing_fraction = sum(f >= _PER_SAMPLE_VALIDITY_THRESHOLD for f in fractions) / len(fractions)

    assert passing_fraction >= _MIN_FRACTION_SAMPLES_PASSING, (
        f"Only {passing_fraction:.1%} of held-out samples ({len(fractions)} total) cleared the "
        f"per-sample {_PER_SAMPLE_VALIDITY_THRESHOLD:.0%} grammar-validity threshold (required: "
        f"{_MIN_FRACTION_SAMPLES_PASSING:.0%}). Either the CLI pipeline is broken, the training "
        f"budget is too small, or a recent change regressed generalization."
    )


def test_grammar_cli_greedy_output_is_fully_grammar_valid(
    grammar_generated_trajectories_greedy: Path, grammar_raw_meds: tuple[Path, Path]
):
    """Strict correctness: greedy (``do_sample=False``) CLI output must be 100% grammar-valid.

    Methodology matches the in-process strictness: for each held-out (subject, trajectory) pair,
    pre-advance a :class:`GrammarFSM` through the subject's prompt tokens, then walk every
    generated grammar token through it. ``valid == total`` on every single sample — no
    passing-rate averaging, no tolerance. If any generated token is an invalid continuation from
    the known prompt state, the test fails with the offending subject, the prompt tokens, and
    the generated tokens up to and including the first invalid one.
    """
    raw_meds_dir, task_labels_dir = grammar_raw_meds
    by_split = _load_trajectories_by_split(grammar_generated_trajectories_greedy)

    prompts_by_subject = prompt_grammar_tokens_by_subject(raw_meds_dir, task_labels_dir, held_out_split)

    total_samples = 0
    failures: list[str] = []
    for t, df in by_split[held_out_split].items():
        tokens_by_subject = grammar_tokens_from_output_df(df)
        for subject_id, tokens in tokens_by_subject.items():
            if not tokens:
                continue
            total_samples += 1
            prompt_tokens = prompts_by_subject[subject_id]
            valid, total = _walk_generated_strictly(prompt_tokens, tokens)
            if valid != total:
                failures.append(
                    f"traj={t} subject={subject_id}: {valid}/{total} valid "
                    f"(first invalid at generated position {valid}): "
                    f"prompt_tokens={prompt_tokens}, generated={tokens}"
                )

    assert total_samples > 0, (
        "No held-out greedy samples had any generated grammar tokens — the model either "
        "produced no output or produced only non-grammar codes."
    )
    assert not failures, (
        f"Greedy CLI output was not 100% grammar-valid on {len(failures)} / {total_samples} "
        f"held-out samples:\n" + "\n".join(failures)
    )


def test_grammar_cli_rolling_actually_rolls_and_preserves_signal(
    grammar_generated_trajectories: Path,
    grammar_raw_meds: tuple[Path, Path],
    grammar_max_seq_len: int,
    grammar_rolling_max_new_tokens: int,
):
    """Rolling-path specific content check.

    The fixture requests ``rolling_generation.max_new_tokens > max_seq_len``, so at least one
    generated sample must have more new tokens than a single HF generate call could produce —
    otherwise rolling wasn't actually exercised even though the fixture asked for it, and any
    signal-preservation claim about the rolling path is vacuous.

    We assert two things:

    1. **Rolling happened.** Some held-out sample's (prompt + new) length exceeds
       ``grammar_max_seq_len``.
    2. **Signal survives rolling.** The rolling subset hits the same per-sample grammar-validity
       threshold as the aggregate signal test. Regresses loudly if sliding-window boundary
       handling corrupts generation.
    """
    raw_meds_dir, task_labels_dir = grammar_raw_meds
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    prompts_by_subject = prompt_grammar_tokens_by_subject(raw_meds_dir, task_labels_dir, held_out_split)

    rolling_sample_lengths: list[int] = []
    rolling_validity_fractions: list[float] = []

    for _t, df in by_split[held_out_split].items():
        sorted_df = df.sort(["subject_id", "time"])
        for subject_id_tup, subject_df in sorted_df.group_by("subject_id", maintain_order=True):
            subject_id = subject_id_tup[0]
            total_len = len(subject_df)
            if total_len <= grammar_max_seq_len:
                continue
            rolling_sample_lengths.append(total_len)
            codes = subject_df["code"].to_list()
            tokens = [CODE_TO_TOKEN[c] for c in codes if c in CODE_TO_TOKEN]
            if not tokens:
                continue
            valid, total = _walk_generated_strictly(prompts_by_subject[subject_id], tokens)
            if total == 0:
                continue
            rolling_validity_fractions.append(valid / total)

    assert rolling_sample_lengths, (
        f"No held-out sample exceeded {grammar_max_seq_len} total tokens even though the "
        f"fixture requested rolling_generation.max_new_tokens={grammar_rolling_max_new_tokens}. "
        f"Either rolling isn't engaged or every subject hit EOS inside a single chunk — the "
        f"rolling path is untested under this fixture."
    )
    assert rolling_validity_fractions, (
        "Samples that rolled were all shorter than the grammar-token floor after filtering out "
        "TIMELINE//* aux codes. The rolling path ran but produced nothing assessable."
    )
    rolling_passing = sum(f >= _PER_SAMPLE_VALIDITY_THRESHOLD for f in rolling_validity_fractions) / len(
        rolling_validity_fractions
    )
    assert rolling_passing >= _MIN_FRACTION_SAMPLES_PASSING, (
        f"Only {rolling_passing:.1%} of rolling held-out samples cleared the "
        f"{_PER_SAMPLE_VALIDITY_THRESHOLD:.0%} per-sample threshold (required: "
        f"{_MIN_FRACTION_SAMPLES_PASSING:.0%}). Rolling happened ({len(rolling_sample_lengths)} "
        f"samples exceeded {grammar_max_seq_len} tokens) but grammar signal didn't survive it — "
        f"suspect the sliding-window boundary handling."
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
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory,
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
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory,
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
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory,
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

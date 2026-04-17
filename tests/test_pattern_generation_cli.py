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
    CODE_TO_TOKEN,
    GRAMMAR_CODES,
    GRAMMAR_MAX_SEQ_LEN,
    GRAMMAR_N_TRAJECTORIES,
    GRAMMAR_ROLLING_MAX_NEW_TOKENS,
    GrammarFSM,
    grammar_tokens_from_output_df,
)
from tests.test_pattern_generation import SEP

# Auxiliary codes that MEICAR_process_data emits for us (timestamps → time-delta tokens; timeline
# start/end sentinels). These aren't part of the grammar but are expected in the output and must
# not cause a grammar-check failure. ``GRAMMAR//`` is **not** listed here: grammar codes must pass
# the exact-membership check against ``GRAMMAR_CODES`` so a stray ``GRAMMAR//bogus`` code can't
# slip through and mask a token→code mapping bug.
_AUX_CODE_PREFIXES = ("TIMELINE//",)


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

    for split, trajectories in by_split.items():
        assert set(trajectories.keys()) == {str(i) for i in range(GRAMMAR_N_TRAJECTORIES)}, (
            f"Split {split}: expected parquets 0..{GRAMMAR_N_TRAJECTORIES - 1}, got {sorted(trajectories)}"
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
        # Must have the same subject set across all trajectories; otherwise ``missing`` would
        # silently count as ``None`` and make a dropped subject look "distinct". The
        # structural test above covers this too, but keeping the assertion here makes this
        # test self-contained.
        for t, per_subject in per_traj_per_subject.items():
            assert set(per_subject.keys()) == subjects_any, (
                f"Split {split} trajectory {t} has subjects {set(per_subject.keys())} but "
                f"trajectory 0 has {subjects_any} — the demux dropped or added subjects."
            )

        # At least one subject must have at least two distinct trajectories out of N.
        any_subject_is_diverse = any(
            len({per_traj_per_subject[t][s] for t in per_traj_per_subject}) > 1 for s in subjects_any
        )
        assert any_subject_is_diverse, (
            f"Split {split}: every subject produced identical trajectories across all "
            f"{GRAMMAR_N_TRAJECTORIES} samples. Either sampling is degenerate (model always "
            f"picks the same token) or the interleaved-predict path collapsed the N "
            f"trajectories into 1."
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
    """Return (n_valid_transitions, n_total_transitions) over the suffix starting at the first SEP.

    Generated trajectories start partway through some program — whatever state the prompt's
    timeline left the model in. A fresh :class:`GrammarFSM` (initial state = BETWEEN) can't judge
    those initial in-program tokens because it would expect a program-start token there. Instead,
    we find the first ``SEP`` token in the generated sequence (which unambiguously returns the
    FSM to BETWEEN state) and score the suffix starting right after it. If no ``SEP`` appears,
    we walk from BETWEEN over the whole sequence — the worst-case interpretation, but the model
    failing to emit *any* SEP in N generated tokens is itself a legitimate negative signal.

    A transition is "valid" if it doesn't send the FSM to the invalid sink; we stop counting at
    the first invalid step, so the return is the valid-prefix length of the scored suffix.
    """
    if SEP in tokens:
        suffix_start = tokens.index(SEP) + 1
        scored = tokens[suffix_start:]
    else:
        scored = tokens

    if not scored:
        return 0, 0

    fsm = GrammarFSM()
    valid = 0
    for tok in scored:
        if fsm.step(tok) is None:
            break
        valid += 1
    return valid, len(scored)


#: A sample must have at least this many grammar tokens before it's scored at all. Must be >= 1
#: (0-length samples have no transitions to validate) but deliberately not higher: the CLI
#: training budget often produces short outputs, and a higher floor would filter out the bulk
#: of the data. The passing-rate metric below is robust to short samples (they either pass or
#: fail the per-sample threshold on their own); a mean-based metric would require a higher
#: floor to avoid being dominated by short-sample outliers.
_MIN_GRAMMAR_TOKENS_FOR_SCORING: int = 1

#: Per-sample threshold: a sample is "passing" if this fraction of its grammar-token transitions
#: are FSM-valid (after stripping the pre-first-SEP prefix; see the walker's docstring). 0.50 is
#: chosen so "passing" means the majority of a sample's emitted grammar tokens continue the
#: grammar — well above anything explainable by chance (a random vocab pick gives ~6% per-step
#: valid rate; the expected prefix-length fraction on long sequences is far below 0.10). The
#: in-process test reaches ~0.9 routinely, so the CLI budget calibrated in the fixture should
#: reach this threshold comfortably.
_PER_SAMPLE_VALIDITY_THRESHOLD: float = 0.50

#: Minimum fraction of held-out (subject, trajectory) samples that must clear the per-sample
#: threshold. Using a passing-rate distributional metric (instead of "best seen") rules out the
#: "one lucky sample masks universal garbage" failure mode a prior reviewer flagged. Short-sample
#: outliers don't drag a passing rate around the way they drag a mean. Tuned empirically on the
#: fixture: the current budget produces ~80% passing rate on held-out at 0.50 per-sample
#: validity, so 0.50 leaves ample sampling-variance headroom while still being a majority
#: requirement.
_MIN_FRACTION_SAMPLES_PASSING: float = 0.50


def test_grammar_cli_model_shows_grammar_signal(grammar_generated_trajectories: Path):
    """Distributional grammar-adherence check on the held-out split.

    This is a **signal** test, not a correctness test. Strict per-token grammar validity is
    covered by ``test_pattern_generation.py``'s in-process tests, which have more training
    budget. Here the CLI budget is fixed, so we verify the model learned something grammatical
    on unseen subjects — which rules out entire classes of CLI bugs (wrong vocab indexing,
    off-by-one in token → code mapping, etc.) that would leave the model emitting
    structurally-random output — while keeping the test robust across minor CLI changes.

    Per-sample metric: (# of valid FSM transitions) / (# of grammar tokens). Walks the sequence
    through :class:`GrammarFSM`; stops at the first invalid step and records how far it got.

    Distributional assertion (not just "best seen"): at least
    ``_MIN_FRACTION_SAMPLES_PASSING`` of held-out (subject, trajectory) pairs clear
    ``_PER_SAMPLE_VALIDITY_THRESHOLD``. This rules out the "one lucky sample masks universal
    garbage" failure mode of a naive "best seen" test while being robust to the
    short-sample outliers that would destabilize a mean-based metric.

    Samples with fewer than ``_MIN_GRAMMAR_TOKENS_FOR_SCORING`` tokens are excluded — they're too
    short for the fraction to be a meaningful distributional signal.
    """
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    # Restrict to held-out: train-split trajectories could exhibit apparent grammar signal from
    # memorization, which wouldn't prove the model generalized. Held-out subjects were never seen
    # during pretraining, so grammar adherence there is real learned signal.
    fractions: list[float] = []
    for _t, df in by_split[held_out_split].items():
        tokens_by_subject = grammar_tokens_from_output_df(df)
        for _subject_id, tokens in tokens_by_subject.items():
            if len(tokens) < _MIN_GRAMMAR_TOKENS_FOR_SCORING:
                continue
            valid, total = _walk_grammar_tokens_ignoring_start(tokens)
            if total == 0:
                continue  # all-SEP edge case — nothing to score after the split
            fractions.append(valid / total)

    assert fractions, (
        f"No held-out samples had >= {_MIN_GRAMMAR_TOKENS_FOR_SCORING} grammar tokens. The "
        f"fixture either generates nothing or generates only non-grammar codes — both indicate "
        f"a broken pipeline."
    )
    passing_fraction = sum(f >= _PER_SAMPLE_VALIDITY_THRESHOLD for f in fractions) / len(fractions)

    assert passing_fraction >= _MIN_FRACTION_SAMPLES_PASSING, (
        f"Only {passing_fraction:.1%} of held-out samples ({len(fractions)} total) cleared the "
        f"per-sample {_PER_SAMPLE_VALIDITY_THRESHOLD:.0%} grammar-validity threshold (required: "
        f"{_MIN_FRACTION_SAMPLES_PASSING:.0%}). Either the CLI pipeline is broken, the training "
        f"budget is too small, or a recent change regressed generalization."
    )


def test_grammar_cli_rolling_actually_rolls_and_preserves_signal(
    grammar_generated_trajectories: Path,
):
    """Rolling-path specific content check.

    The fixture sets ``rolling_generation.max_new_tokens=GRAMMAR_ROLLING_MAX_NEW_TOKENS`` (> the
    model's ``max_seq_len``), so **at least one generated sample must have more new tokens than
    a single HF generate call could produce** — otherwise rolling wasn't actually exercised even
    though the fixture asked for it, and any signal-preservation claim about the rolling path is
    vacuous.

    We assert two things:

    1. **Rolling happened.** Some held-out sample's (prompt + new) length exceeds
       ``GRAMMAR_MAX_SEQ_LEN`` — a single-chunk call is capped at ``max_seq_len`` tokens total,
       so exceeding that proves the sliding-window loop appended new tokens past the single-
       chunk ceiling.
    2. **Signal survives rolling.** Among the samples that rolled, at least a quarter still hit
       the per-sample grammar-validity threshold. If rolling-boundary handling were broken (e.g.
       positional-embedding discontinuity at the slide, missing EOS carry-over, demux corruption
       across chunks), rolling samples would be *worse* than non-rolling samples. A complementary
       signal check on just those samples catches regressions that the aggregate signal test
       above would miss if non-rolling samples dominated the mean.
    """
    by_split = _load_trajectories_by_split(grammar_generated_trajectories)

    rolling_sample_lengths: list[int] = []
    rolling_validity_fractions: list[float] = []

    for _t, df in by_split[held_out_split].items():
        sorted_df = df.sort(["subject_id", "time"])
        # Per-subject total length (prompt + new). A MEDS event row has a subject_id, so
        # ``len(subject_df)`` directly equals the number of tokens emitted for that subject.
        for _subject_id, subject_df in sorted_df.group_by("subject_id", maintain_order=True):
            total_len = len(subject_df)
            if total_len <= GRAMMAR_MAX_SEQ_LEN:
                continue
            rolling_sample_lengths.append(total_len)
            codes = subject_df["code"].to_list()
            tokens = [CODE_TO_TOKEN[c] for c in codes if c in CODE_TO_TOKEN]
            if len(tokens) < _MIN_GRAMMAR_TOKENS_FOR_SCORING:
                continue
            valid, total = _walk_grammar_tokens_ignoring_start(tokens)
            if total == 0:
                continue
            rolling_validity_fractions.append(valid / total)

    assert rolling_sample_lengths, (
        f"No held-out sample exceeded {GRAMMAR_MAX_SEQ_LEN} total tokens even though the fixture "
        f"requested rolling_generation.max_new_tokens={GRAMMAR_ROLLING_MAX_NEW_TOKENS}. Either "
        f"rolling isn't actually engaged or every subject hit EOS inside a single chunk — the "
        f"rolling path is untested under this fixture."
    )
    assert rolling_validity_fractions, (
        "Samples that rolled were all shorter than the grammar-token floor after filtering out "
        "TIMELINE//* aux codes. The rolling path ran but produced nothing assessable."
    )
    rolling_passing = sum(f >= _PER_SAMPLE_VALIDITY_THRESHOLD for f in rolling_validity_fractions) / len(
        rolling_validity_fractions
    )
    # Same threshold as the aggregate signal test: the rolling path shouldn't be *worse* than the
    # non-rolling subset of the same distribution, so we expect the same passing rate to hold.
    assert rolling_passing >= _MIN_FRACTION_SAMPLES_PASSING, (
        f"Only {rolling_passing:.1%} of rolling held-out samples cleared the "
        f"{_PER_SAMPLE_VALIDITY_THRESHOLD:.0%} per-sample threshold (required: "
        f"{_MIN_FRACTION_SAMPLES_PASSING:.0%}). Rolling happened ({len(rolling_sample_lengths)} "
        f"samples exceeded {GRAMMAR_MAX_SEQ_LEN} tokens) but grammar signal didn't survive it — "
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

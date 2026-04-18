"""Train a tiny ``Model`` on the grammar in-process and verify generation stays grammar-valid.

All other tests in this repo either exercise the generation code path on a random-init model
(doctests, ``test_generate_trajectories.py``'s direct unit tests) or drive the full CLI (the
end-to-end grammar test at ``tests/grammar/test_cli.py``). This file fills the middle: a
generation-correctness test that (1) trains a tiny ``Model`` on CPU in a handful of seconds, (2)
has unambiguous ground truth because the training distribution is a finite-state grammar, and
(3) asserts grammar adherence via a deterministic FSM walk of the generated tokens. The grammar
definition, FSM, and training-sample sampler live in :mod:`tests.grammar._grammar` — see that
module's docstring for the grammar specification.
"""

from __future__ import annotations

import random

import pytest
import torch

from MEDS_EIC_AR.model.model import Model
from tests.grammar._grammar import (
    ALT_PROGRAMS,
    DUMMY_EOS,
    MAX_SEQ_LEN,
    PAD,
    PROGRAMS,
    SEP,
    VOCAB_SIZE,
    GrammarFSM,
    build_training_batch_codes,
    mock_batch,
)

# ---------------------------------------------------------------------------
# Trained model fixture
# ---------------------------------------------------------------------------


_NUM_TRAIN_STEPS = 400
_BATCH_SIZE = 32
_LEARNING_RATE = 3e-3


@pytest.fixture(scope="module")
def grammar_trained_model() -> Model:
    """Train a tiny ``Model`` on the pattern grammar.

    Module-scoped so both tests share it.
    """
    torch.manual_seed(0)
    rng = random.Random(0)

    model = Model(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "max_position_embeddings": MAX_SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "eos_token_id": DUMMY_EOS,
        },
        precision="32-true",
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)

    for _ in range(_NUM_TRAIN_STEPS):
        batch_codes = build_training_batch_codes(rng, _BATCH_SIZE)
        batch = mock_batch(batch_codes)
        optimizer.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# FSM negative-control unit tests (no training, very fast)
# ---------------------------------------------------------------------------


def test_grammar_fsm_rejects_hand_crafted_invalid_sequences():
    """Negative control: feed the FSM sequences we know are invalid and assert it rejects each at the
    expected position.

    Without this test, an always-accepting FSM (e.g., one with a bug in the internal transition
    logic) would let the downstream generation-correctness tests pass vacuously — a trained model
    could emit garbage and still appear grammar-valid. Each case here exercises a distinct
    failure mode of the FSM.
    """
    # (1) Wrong token at BETWEEN: feeding a non-start token rejects immediately.
    assert GrammarFSM().walk([SEP]) == 0
    assert GrammarFSM().walk([PAD]) == 0
    assert GrammarFSM().walk([DUMMY_EOS]) == 0  # out-of-grammar dummy
    # (2) Skipping a token inside a program (A_start followed by A[2] instead of A[1]).
    assert GrammarFSM().walk([PROGRAMS["A"][0], PROGRAMS["A"][2]]) == 1
    # (3) Wrong token at the tail of a program (complete A then emit A[0] instead of SEP).
    tokens = [*PROGRAMS["A"], PROGRAMS["A"][0]]  # A A A A A_start  → 2,3,4,5,2
    assert GrammarFSM().walk(tokens) == 4
    # (4) Consecutive separators: SEP right after another SEP is a BETWEEN-state violation.
    tokens = [*PROGRAMS["A"], SEP, SEP]
    assert GrammarFSM().walk(tokens) == 5
    # (5) Switching programs mid-stream (A_start followed by B[1], skipping all of A).
    assert GrammarFSM().walk([PROGRAMS["A"][0], PROGRAMS["B"][1]]) == 1
    # (6) Valid program followed by wrong continuation after SEP.
    tokens = [*PROGRAMS["A"], SEP, PROGRAMS["A"][1]]  # "A | A[1]" — SEP expects a start
    assert GrammarFSM().walk(tokens) == 5


def test_grammar_fsm_alt_grammar_rejects_default_grammar_sequences():
    """Symmetric negative control at the grammar level: confirm ``PROGRAMS`` and ``ALT_PROGRAMS`` actually
    disagree on the same input sequences.

    If both FSMs accepted the same sequences, the cross-grammar assertion in
    ``test_trained_model_rolling_generation_preserves_grammar`` would be vacuous. This test
    proves the two grammars are materially different: a hand-built ``PROGRAMS``-valid sequence
    is rejected by the ``ALT_PROGRAMS`` FSM and vice versa.
    """
    programs_valid = [*PROGRAMS["A"], SEP, *PROGRAMS["B"], SEP]
    alt_valid = [*ALT_PROGRAMS["A"], SEP, *ALT_PROGRAMS["B"], SEP]

    assert GrammarFSM(programs=PROGRAMS).walk(programs_valid) == len(programs_valid)
    assert GrammarFSM(programs=ALT_PROGRAMS).walk(alt_valid) == len(alt_valid)
    assert GrammarFSM(programs=PROGRAMS).walk(alt_valid) < len(alt_valid)
    assert GrammarFSM(programs=ALT_PROGRAMS).walk(programs_valid) < len(programs_valid)


# ---------------------------------------------------------------------------
# Generation-correctness tests against the trained model
# ---------------------------------------------------------------------------


_PROMPTS_AND_EXPECTED_COMPLETIONS: list[tuple[list[int], list[int]]] = [
    # ``A`` prompt: start of program A. Must continue with A[1], A[2], A[3].
    ([PROGRAMS["A"][0]], list(PROGRAMS["A"][1:])),
    # Mid-program: first two tokens of A. Must continue with A[2], A[3].
    (list(PROGRAMS["A"][:2]), list(PROGRAMS["A"][2:])),
    # ``B`` prompt.
    ([PROGRAMS["B"][0]], list(PROGRAMS["B"][1:])),
    # ``C`` prompt — the long one. Tests that the model has learned the full 9-token sequence.
    ([PROGRAMS["C"][0]], list(PROGRAMS["C"][1:])),
    # Mid-C: enough context to pin down the completion.
    (list(PROGRAMS["C"][:3]), list(PROGRAMS["C"][3:])),
]


def test_trained_model_single_chunk_generation_recovers_grammar(grammar_trained_model: Model):
    """For each prompt, greedy single-chunk generation must complete the in-progress program and then emit
    ``SEP`` as the transition token.

    ``SEP`` is the **only** valid transition out of the final position of a program under this
    grammar: the training distribution never contains an EOS token (see the module docstring),
    so the model cannot have learned to emit anything else at that position.

    Generation uses ``do_sample=False`` so the model greedy-decodes each next token (argmax over
    logits, no temperature sampling). That makes the output fully deterministic given the trained
    weights and the prompt, so a one-shot assertion against the expected program completion is
    meaningful — we're testing "does the model's argmax follow the grammar?", not "does some
    stochastic sample happen to do so."
    """
    model = grammar_trained_model

    for prompt, expected_completion in _PROMPTS_AND_EXPECTED_COMPLETIONS:
        prompt_tensor = torch.tensor([prompt], dtype=torch.long)
        batch = mock_batch(prompt_tensor)

        generated = model.generate(batch, do_sample=False)  # single-chunk path
        produced = generated[0].tolist()

        check_len = len(expected_completion) + 1
        assert len(produced) >= check_len, (
            f"Expected at least {check_len} generated tokens for prompt {prompt}, "
            f"got {len(produced)}: {produced}."
        )
        assert produced[: len(expected_completion)] == expected_completion, (
            f"Greedy completion of prompt {prompt} should be {expected_completion}; "
            f"model produced {produced[: len(expected_completion)]}."
        )

        transition_token = produced[len(expected_completion)]
        assert transition_token == SEP, (
            f"After completing prompt {prompt}, expected SEP={SEP} as the transition token; "
            f"got {transition_token}. Full output: {produced}."
        )


_ROLLING_BUDGET = 50


def test_trained_model_rolling_generation_preserves_grammar(grammar_trained_model: Model):
    """Rolling generation past the model's context window must stay grammar-valid.

    The trained model has ``max_position_embeddings=16`` and we generate
    ``max_new_tokens=_ROLLING_BUDGET`` from a short ``[A_start]`` prompt. The budget is set
    strictly above ``max_seq_len`` so the rolling loop is guaranteed to cross at least one
    sliding-window boundary regardless of when the model decides to stop. We then walk every
    generated token through the grammar FSM and assert each transition is valid.

    **The contract we're testing is semantic, not structural.** ``_rolling_generate`` promises
    "up to ``max_new_tokens`` tokens, possibly fewer on EOS" — not "exactly that many," and not
    any particular inner-call count. So the two assertions below are:

    1. Every generated token is a valid FSM transition (the grammar-correctness invariant).
    2. The continuation is longer than a single non-rolling chunk could have produced from this
       prompt, which proves the rolling path was actually exercised.
    """
    model = grammar_trained_model

    prompt = [PROGRAMS["A"][0]]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long)
    batch = mock_batch(prompt_tensor)

    generated = model.generate(batch, do_sample=False, max_new_tokens=_ROLLING_BUDGET)
    produced = generated[0].tolist()

    single_chunk_cap = MAX_SEQ_LEN - len(prompt)
    assert len(produced) > single_chunk_cap, (
        f"Rolling generation produced {len(produced)} tokens from a {len(prompt)}-token prompt; "
        f"that's within the single-chunk cap of {single_chunk_cap} (= max_seq_len - len(prompt)), "
        f"so the rolling path may not have been exercised. Full output: {produced}."
    )

    fsm = GrammarFSM()
    for tok in prompt:
        fsm.step(tok)
    assert fsm.state == ("IN", "A", 0), (
        f"Expected FSM in ('IN', 'A', 0) after prompt {prompt}, got {fsm.state}."
    )

    first_invalid = fsm.walk(produced)
    assert first_invalid == len(produced), (
        f"Rolling generation emitted a grammar-invalid token at position {first_invalid} "
        f"(token={produced[first_invalid]}). FSM state just before the violation: "
        f"{fsm.state}. Full output: {produced}."
    )

    # Negative control (cross-grammar): the same continuation must be rejected by ALT_PROGRAMS
    # within a very short prefix.
    alt_fsm = GrammarFSM(programs=ALT_PROGRAMS)
    for tok in prompt:
        alt_fsm.step(tok)
    alt_first_invalid = alt_fsm.walk(produced)
    assert alt_first_invalid < len(produced), (
        f"ALT_PROGRAMS FSM unexpectedly accepted the entire continuation. The FSM may be "
        f"degenerate (always-accept bug) or the trained model happens to emit an ALT-valid "
        f"sequence (statistically implausible for seeded greedy decoding on 50 tokens). "
        f"Full output: {produced}."
    )

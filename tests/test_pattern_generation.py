"""Integration test: train a tiny ``Model`` on a synthetic pattern grammar and verify that
generation — both single-chunk and rolling — recovers the grammar.

All other tests in this repo either exercise the generation code path on a random-init model
(doctests, ``test_generate_trajectories.py``'s direct unit tests) or drive the full CLI but only
check parquet shape. This file fills the missing middle: a real generation-correctness test that
(1) trains a tiny ``Model`` on CPU in a handful of seconds, (2) has unambiguous ground truth
because the training distribution is a finite-state grammar, and (3) asserts grammar adherence
via a deterministic FSM walk of the generated tokens.

The grammar has three programs — ``A B C D``, ``1 2 3``, ``R S T U V W X Y Z`` — emitted
stochastically with a ``|`` separator between them. Training sequences are packed full of
programs until no more fit, then padded — **no terminating EOS token**. That's intentional: if
we trained the model to emit EOS, the rolling-generation test would terminate early (the model
would emit EOS before the sliding window had a chance to cross a boundary) and the real
integration signal — "rolling generation stays grammar-valid across multiple sliding windows" —
would be smothered by early termination. Instead, we set the model's ``eos_token_id`` to an
out-of-grammar dummy token and let ``max_new_tokens`` control termination for the rolling test.

The whole file fits in under a minute of CPU training on a small model and uses no MEDS
fixtures; it talks to ``Model`` directly via mock batches.

## What the training data looks like

The module-level doctest below shows concretely what a training sequence and a training batch
look like, so anyone reading this file can verify the grammar encoding by inspection rather than
having to trust the FSM implementation.

    >>> import random, torch
    >>> _ = torch.manual_seed(0)

Token table::

    PAD = 0         # pad token, excluded from loss via ignore_index
    SEP = 1         # between programs
    A   = (2, 3, 4, 5)              # program A — four tokens
    B   = (6, 7, 8)                 # program B — three tokens
    C   = (9, 10, 11, 12, 13, 14, 15, 16, 17)  # program C — nine tokens
    DUMMY_EOS = 18  # reserved out-of-grammar EOS the model should never produce

One training sequence is a run of programs separated by ``SEP``, packed until the next program
wouldn't fit. With a seeded RNG the first sequence under ``max_len=16`` is::

    >>> rng = random.Random(0)
    >>> seq = sample_sequence(rng, max_len=16)
    >>> seq
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 1]

Read left-to-right: program C (9, 10, ..., 17), then SEP. The RNG sampled program C first, then
program A was going to be next but wouldn't fit (4 more tokens + SEP = 5 > 16 - 10 = 6... actually
*would* fit, but the sampler picked a different program next that also wouldn't fit, and the
packing loop bails out rather than retrying). ``GrammarFSM`` accepts the full sequence::

    >>> GrammarFSM().walk(seq) == len(seq)
    True

A training batch stacks several such sequences and right-pads to the longest::

    >>> rng = random.Random(0)
    >>> batch = build_training_batch_codes(rng, batch_size=3)
    >>> batch
    tensor([[ 9, 10, 11, 12, 13, 14, 15, 16, 17,  1,  0,  0,  0,  0,  0],
            [ 6,  7,  8,  1,  2,  3,  4,  5,  1,  6,  7,  8,  1,  0,  0],
            [ 9, 10, 11, 12, 13, 14, 15, 16, 17,  1,  2,  3,  4,  5,  1]])

Row 1 is ``C | <pad>``, row 2 is ``B | A | B | <pad>``, row 3 is ``C | A`` — all valid under the
grammar, all right-padded with ``PAD=0`` where they ran short. ``Model._forward``'s cross-entropy
loss ignores those pad positions so the model never learns "what comes after ``PAD``."

## Negative controls

To protect against "always-accepting FSM" bugs that would make the generation-correctness tests
vacuous, the file also carries a second grammar ``ALT_PROGRAMS`` — same start tokens and same
token sets as ``PROGRAMS`` but with the internal order of each program permuted — and asserts
that ``PROGRAMS`` and ``ALT_PROGRAMS`` disagree on hand-built canonical sequences.

Concretely, program A is ``(2, 3, 4, 5)`` under ``PROGRAMS`` and ``(2, 4, 3, 5)`` under
``ALT_PROGRAMS``. A canonical "A then SEP" sequence written in the ``PROGRAMS`` ordering is
accepted by the default FSM and rejected by the ALT FSM, and vice versa::

    >>> programs_A = [2, 3, 4, 5, 1]  # valid under PROGRAMS
    >>> alt_A      = [2, 4, 3, 5, 1]  # valid under ALT_PROGRAMS
    >>> GrammarFSM(programs=PROGRAMS).walk(programs_A)
    5
    >>> GrammarFSM(programs=ALT_PROGRAMS).walk(programs_A)
    1
    >>> GrammarFSM(programs=ALT_PROGRAMS).walk(alt_A)
    5
    >>> GrammarFSM(programs=PROGRAMS).walk(alt_A)
    1

The ``1`` return values in rows 2 and 4 are the crux: each FSM rejects the other's canonical
sequence at position 1 (the first internal program transition, where the two grammars disagree
on what should follow A[0]=2). The generation-correctness test then additionally asserts that
the trained model's output is **rejected** by ``ALT_PROGRAMS``'s FSM within a short prefix,
proving the trained model learned ``PROGRAMS`` specifically rather than being a uniform-random
generator.
"""

from __future__ import annotations

import random
from unittest.mock import Mock

import pytest
import torch

from MEDS_EIC_AR.model.model import Model

# ---------------------------------------------------------------------------
# Grammar vocabulary
# ---------------------------------------------------------------------------

PAD = 0
SEP = 1
# Three programs, each a fixed token sequence. Tokens are picked to be mnemonic: A..D as 2..5 (the
# four-token program, analogous to "A B C D"), 1..3 as 6..8 (the three-token program), and R..Z as
# 9..17 (the nine-token program).
PROGRAMS: dict[str, tuple[int, ...]] = {
    "A": (2, 3, 4, 5),
    "B": (6, 7, 8),
    "C": (9, 10, 11, 12, 13, 14, 15, 16, 17),
}
PROGRAM_NAMES = tuple(PROGRAMS.keys())
PROGRAM_WEIGHTS = (0.4, 0.3, 0.3)
PROGRAM_STARTS = {name: prog[0] for name, prog in PROGRAMS.items()}

# **Negative-control alternative grammar.** Same start tokens as ``PROGRAMS``, same token sets per
# program, but the *order* within each program is different: pairs of adjacent positions are
# swapped. A model trained on ``PROGRAMS`` must produce sequences that a FSM over ``ALT_PROGRAMS``
# rejects within a very short prefix — the trained model's "A_start followed by A[1]=3" transition
# violates ``ALT_PROGRAMS["A"]``'s expected "A_start followed by 4". We use this to prove both that
# the FSM actually rejects (not always-accepts) and that the trained model learned ``PROGRAMS``
# specifically rather than any uniform-random generator.
ALT_PROGRAMS: dict[str, tuple[int, ...]] = {
    "A": (2, 4, 3, 5),
    "B": (6, 8, 7),
    "C": (9, 11, 10, 13, 12, 15, 14, 17, 16),
}

# Reserved dummy EOS index outside the training distribution. The model is configured with
# ``eos_token_id=DUMMY_EOS`` so the rolling-generation validation (which rejects ``None`` and pad
# collisions) is satisfied, but the model is never trained on sequences containing ``DUMMY_EOS`` so
# it will never emit this token greedily. Rolling generation is bounded solely by ``max_new_tokens``.
DUMMY_EOS = 18
VOCAB_SIZE = 20
MAX_SEQ_LEN = 16


def _start_token_to_name(token: int, programs: dict[str, tuple[int, ...]] = PROGRAMS) -> str | None:
    for name, prog in programs.items():
        if token == prog[0]:
            return name
    return None


# ---------------------------------------------------------------------------
# Grammar FSM
# ---------------------------------------------------------------------------


class GrammarFSM:
    """Deterministic walker over a pattern grammar. Each ``step(token)`` returns either the new state or
    ``None`` (meaning the transition is grammar-invalid).

    Defaults to the ``PROGRAMS`` grammar; pass ``programs=ALT_PROGRAMS`` to walk the
    negative-control grammar instead. Both grammars share the same structural shape — three
    named programs, a ``SEP`` between them — so the same FSM logic works for both.

    States:
      - ``"BETWEEN"`` — between programs (or at the very start). Valid next tokens: any
        program-start under the active ``programs`` dict.
      - ``("IN", prog_name, pos)`` — inside program ``prog_name`` having just emitted position
        ``pos``. Valid next: either ``programs[prog_name][pos + 1]`` if there is one, or (at the
        final position) ``SEP``.

    No terminal state: the training distribution never emits an EOS token (see module
    docstring), so the grammar is effectively infinite — any valid sequence of programs
    separated by ``SEP`` can continue indefinitely.

    >>> fsm = GrammarFSM()
    >>> fsm.step(2)  # start of program A
    ('IN', 'A', 0)
    >>> fsm.step(3)
    ('IN', 'A', 1)
    >>> fsm.step(4)
    ('IN', 'A', 2)
    >>> fsm.step(5)
    ('IN', 'A', 3)
    >>> fsm.step(SEP)
    'BETWEEN'
    >>> fsm.step(6)  # start of program B
    ('IN', 'B', 0)
    >>> _ = fsm.step(8)  # would skip B[1]=7, invalid — assign to _ so doctest doesn't print None
    >>> fsm.state is None
    True

    Under the alternative grammar the same PROGRAMS-valid continuation is rejected at the first
    internal transition:

    >>> alt = GrammarFSM(programs=ALT_PROGRAMS)
    >>> alt.step(2)  # start of program A in ALT (2, 4, 3, 5)
    ('IN', 'A', 0)
    >>> _ = alt.step(3)  # PROGRAMS expects 3 next; ALT expects 4. Rejected.
    >>> alt.state is None
    True
    """

    def __init__(self, programs: dict[str, tuple[int, ...]] = PROGRAMS) -> None:
        self.programs = programs
        self.state: str | tuple[str, str, int] | None = "BETWEEN"

    def step(self, token: int) -> str | tuple[str, str, int] | None:
        """Transition on ``token``. Returns new state, or ``None`` on invalid transition.

        Once the FSM enters the invalid state (``self.state is None``), further ``step`` calls
        remain in ``None``.
        """
        if self.state is None:
            return None
        if self.state == "BETWEEN":
            name = _start_token_to_name(token, self.programs)
            if name is None:
                self.state = None
            else:
                self.state = ("IN", name, 0)
            return self.state
        # ("IN", name, pos)
        _, name, pos = self.state
        prog = self.programs[name]
        if pos + 1 < len(prog):
            if token == prog[pos + 1]:
                self.state = ("IN", name, pos + 1)
            else:
                self.state = None
            return self.state
        # At last position of program; next must be SEP.
        if token == SEP:
            self.state = "BETWEEN"
        else:
            self.state = None
        return self.state

    def walk(self, tokens) -> int:
        """Walk a token sequence, returning the index of the first invalid token.

        Returns ``len(tokens)`` if every transition was valid (i.e. the entire sequence was
        accepted). This is the primary entry point that the model-generation tests use: after
        running ``Model.generate(...)``, they walk the resulting token list through a freshly
        initialized FSM and assert the return value equals ``len(tokens)``.

        Examples:
            A fully valid PROGRAMS-grammar sequence (program A, then SEP, then program B, then
            SEP) is accepted all the way through — ``walk`` returns the full length:

            >>> tokens = [2, 3, 4, 5, 1, 6, 7, 8, 1]  # A | B |
            >>> GrammarFSM().walk(tokens)
            9
            >>> GrammarFSM().walk(tokens) == len(tokens)
            True

            If we poison the middle of program A by swapping A[1] and A[2], ``walk`` rejects at
            the first offending position (here position 2 — A[0]=2 is fine, expected A[1]=3 but
            we emit 4 instead):

            >>> GrammarFSM().walk([2, 4, 3, 5, 1])
            1

            Emitting the wrong start token at ``BETWEEN`` is rejected at position 0:

            >>> GrammarFSM().walk([SEP])
            0
            >>> GrammarFSM().walk([99])  # not a program start
            0

            The same input is accepted by ``ALT_PROGRAMS`` (whose program A is ``(2, 4, 3, 5)``)
            but rejected by ``PROGRAMS`` (whose program A is ``(2, 3, 4, 5)``):

            >>> seq = [2, 4, 3, 5, 1]  # valid under ALT, invalid under default
            >>> GrammarFSM(programs=ALT_PROGRAMS).walk(seq)
            5
            >>> GrammarFSM(programs=PROGRAMS).walk(seq)
            1
        """
        for i, tok in enumerate(tokens):
            if self.step(int(tok)) is None:
                return i
        return len(tokens)


# ---------------------------------------------------------------------------
# Grammar sampling
# ---------------------------------------------------------------------------


def sample_sequence(rng: random.Random, max_len: int = MAX_SEQ_LEN) -> list[int]:
    """Sample one grammar-valid sequence that fits in ``max_len`` tokens, no terminating EOS.

    Packs programs separated by ``SEP`` until the next full program wouldn't fit. The sequence
    always ends with a ``SEP``; there is no terminating EOS. ``build_training_batch_codes`` pads
    a collection of these sequences to the longest one in the batch with ``PAD``, and
    ``Model._forward``'s cross-entropy loss ignores those pad positions via ``ignore_index``, so
    the model never sees a "what comes after the final SEP" target and the trained grammar
    remains effectively non-terminating.

    >>> rng = random.Random(0)
    >>> seq = sample_sequence(rng)
    >>> seq
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 1]
    >>> # That's program C (9..17) followed by SEP (1).
    >>> GrammarFSM().walk(seq) == len(seq)
    True
    >>> seq[-1] == SEP
    True

    Sampling a few more seeded sequences shows how the grammar packs programs into ``max_len``:

    >>> [sample_sequence(rng) for _ in range(3)]  # doctest: +NORMALIZE_WHITESPACE
    [[6, 7, 8, 1, 2, 3, 4, 5, 1, 6, 7, 8, 1],
     [9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 1],
     [6, 7, 8, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1]]
    """
    tokens: list[int] = []
    while True:
        prog_name = rng.choices(PROGRAM_NAMES, weights=PROGRAM_WEIGHTS, k=1)[0]
        prog = PROGRAMS[prog_name]
        # Reserve room for the SEP after this program.
        if len(tokens) + len(prog) + 1 > max_len:
            break
        tokens.extend(prog)
        tokens.append(SEP)
    if not tokens:  # pragma: no cover — only triggered if max_len is absurdly small
        tokens.extend(PROGRAMS["B"])
        tokens.append(SEP)
    return tokens


def build_training_batch_codes(
    rng: random.Random, batch_size: int, max_len: int = MAX_SEQ_LEN
) -> torch.Tensor:
    """Build a ``[batch_size, L]`` token-code tensor where ``L`` is the longest sequence in the batch.

    Pads shorter sequences with ``PAD`` on the right. Note this pads **to the longest sequence in
    the batch**, not to ``max_len`` — ``Model._forward`` ignores ``PAD`` via ``ignore_index`` in
    cross-entropy so either convention would produce the same loss, but padding only to the batch
    max saves a few forward-pass FLOPs per step.

    Returns the raw token-code tensor. The separate :func:`mock_batch` helper wraps it in a
    minimal ``MEDSTorchBatch``-shaped mock so ``Model.forward`` / ``Model.generate`` can consume
    it. We keep the two steps distinct because training only needs the codes tensor and calling
    ``mock_batch`` inside the training loop would be noise. The ``_codes`` suffix in the name is
    also how we sidestep a collision with the session-scoped ``sample_batch`` fixture that
    ``conftest.py`` injects into the doctest namespace.

    >>> rng = random.Random(0)
    >>> batch_codes = build_training_batch_codes(rng, batch_size=3)
    >>> batch_codes
    tensor([[ 9, 10, 11, 12, 13, 14, 15, 16, 17,  1,  0,  0,  0,  0,  0],
            [ 6,  7,  8,  1,  2,  3,  4,  5,  1,  6,  7,  8,  1,  0,  0],
            [ 9, 10, 11, 12, 13, 14, 15, 16, 17,  1,  2,  3,  4,  5,  1]])

    Row 0 is program C (9..17) then SEP (1) then PAD. Row 1 is programs B, A, B separated by SEPs
    then PAD. Row 2 is C then A — it fills ``max_len=16`` exactly, so no PAD positions. Every
    non-pad row is a valid program-SEP sequence under the grammar, and PAD positions at the right
    are ignored during training via cross-entropy's ``ignore_index``.
    """
    seqs = [sample_sequence(rng, max_len) for _ in range(batch_size)]
    pad_to = max(len(s) for s in seqs)
    padded = torch.full((batch_size, pad_to), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded


def mock_batch(code: torch.Tensor) -> Mock:
    """Build a minimal ``MEDSTorchBatch``-shaped mock that ``Model`` can consume."""
    return Mock(code=code, PAD_INDEX=PAD, mode="SM")


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
# Generation tests
# ---------------------------------------------------------------------------


# For each program, a short prompt that should unambiguously constrain the next several tokens.
# The first token is the program's start; we assert greedy decoding completes the program, then
# transitions via SEP or EOS (both FSM-valid).
# ---------------------------------------------------------------------------
# FSM negative-control unit tests (no training, very fast)
# ---------------------------------------------------------------------------


def test_grammar_fsm_rejects_hand_crafted_invalid_sequences():
    """Negative control: feed the FSM sequences we know are invalid and assert it rejects each
    at the expected position.

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
    """Symmetric negative control at the grammar level: confirm ``PROGRAMS`` and ``ALT_PROGRAMS``
    actually disagree on the same input sequences.

    If both FSMs accepted the same sequences, the cross-grammar assertion in
    ``test_trained_model_rolling_generation_preserves_grammar`` would be vacuous. This test
    proves the two grammars are materially different: a hand-built ``PROGRAMS``-valid sequence
    is rejected by the ``ALT_PROGRAMS`` FSM and vice versa.
    """

    # Canonical PROGRAMS sequence: A then SEP then B then SEP.
    programs_valid = [*PROGRAMS["A"], SEP, *PROGRAMS["B"], SEP]
    # Canonical ALT_PROGRAMS sequence: same shape, alt-ordered programs.
    alt_valid = [*ALT_PROGRAMS["A"], SEP, *ALT_PROGRAMS["B"], SEP]

    # Each grammar accepts its own canonical sequence in full.
    assert GrammarFSM(programs=PROGRAMS).walk(programs_valid) == len(programs_valid)
    assert GrammarFSM(programs=ALT_PROGRAMS).walk(alt_valid) == len(alt_valid)
    # And each grammar rejects the other's canonical sequence at some internal transition.
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

        # do_sample=False → greedy argmax decoding (see test docstring).
        generated = model.generate(batch, do_sample=False)  # single-chunk path
        produced = generated[0].tolist()

        # We only need to check enough tokens to cover the expected completion plus the
        # transition token; anything beyond that is a don't-care here.
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

    Both invariants survive healthy refactors (e.g., swapping the inner generate call for a
    faster backend, or changing how ``rolling_context_size`` is computed) because neither depends
    on the implementation strategy of the rolling loop.
    """
    model = grammar_trained_model

    # Seed with the start of program A.
    prompt = [PROGRAMS["A"][0]]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long)
    batch = mock_batch(prompt_tensor)

    # do_sample=False → greedy argmax decoding, same as the single-chunk test: the output is
    # fully determined by the trained weights and the prompt, so we can assert on specific
    # grammar-FSM transitions rather than on distributions.
    generated = model.generate(batch, do_sample=False, max_new_tokens=_ROLLING_BUDGET)
    produced = generated[0].tolist()

    # (2): length exceeds the single-chunk ceiling of ``max_seq_len - len(prompt)`` new tokens.
    # A non-rolling ``Model.generate`` on the same prompt can emit at most that many tokens (the
    # remainder of the model's context window); anything longer must have come from the rolling
    # path crossing at least one sliding boundary.
    single_chunk_cap = MAX_SEQ_LEN - len(prompt)
    assert len(produced) > single_chunk_cap, (
        f"Rolling generation produced {len(produced)} tokens from a {len(prompt)}-token prompt; "
        f"that's within the single-chunk cap of {single_chunk_cap} (= max_seq_len - len(prompt)), "
        f"so the rolling path may not have been exercised. Full output: {produced}."
    )

    # Replay the prompt through the FSM to establish the starting state for the continuation.
    fsm = GrammarFSM()
    for tok in prompt:
        fsm.step(tok)
    assert fsm.state == ("IN", "A", 0), (
        f"Expected FSM in ('IN', 'A', 0) after prompt {prompt}, got {fsm.state}."
    )

    # (1): walk every emitted token through the FSM. The first invalid transition is a test
    # failure regardless of where in the continuation it happened.
    first_invalid = fsm.walk(produced)
    assert first_invalid == len(produced), (
        f"Rolling generation emitted a grammar-invalid token at position {first_invalid} "
        f"(token={produced[first_invalid]}). FSM state just before the violation: "
        f"{fsm.state}. Full output: {produced}."
    )

    # Negative control (cross-grammar): the same continuation must be rejected by ALT_PROGRAMS
    # within a very short prefix. If it weren't, the FSM would accept any sequence of tokens and
    # the "grammar valid" assertion above would be vacuous. See ``ALT_PROGRAMS`` for the contrast:
    # the trained model emits A[0], A[1]=3 next, but ALT_A's second token is 4, so the walk
    # rejects at position 1.
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

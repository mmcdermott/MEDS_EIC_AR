"""Integration test: train a tiny ``Model`` on a synthetic pattern grammar and verify that
generation — both single-chunk and rolling — recovers the grammar.

All other tests in this repo either exercise the generation code path on a random-init model
(doctests, ``test_generate_trajectories.py``'s direct unit tests) or drive the full CLI but only
check parquet shape. This file fills the missing middle: a real generation-correctness test that
(1) trains a tiny ``Model`` on CPU in a handful of seconds, (2) has unambiguous ground truth because
the training distribution is a finite-state grammar, and (3) asserts grammar adherence via a
deterministic FSM walk of the generated tokens.

The grammar has three programs — ``A B C D``, ``1 2 3``, ``R S T U V W X Y Z`` — emitted
stochastically with a ``|`` separator between them. Training sequences are packed full of programs
until no more fit, then padded — **no terminating EOS token**. That's intentional: if we trained
the model to emit EOS, the rolling-generation test would terminate early (the model would emit EOS
before the sliding window had a chance to cross a boundary) and the real integration signal —
"rolling generation stays grammar-valid across multiple sliding windows" — would be smothered by
early termination. Instead, we set the model's ``eos_token_id`` to an out-of-grammar dummy token
and let ``max_new_tokens`` control termination for the rolling test. The single-chunk test
allows ``SEP`` as the transition token out of a program (which the model will emit, since it's
what the training distribution always does at that position).

The whole file fits in under a minute of CPU training on a small model and uses no MEDS fixtures;
it talks to ``Model`` directly via mock batches.
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

# Reserved dummy EOS index outside the training distribution. The model is configured with
# ``eos_token_id=DUMMY_EOS`` so the rolling-generation validation (which rejects ``None`` and pad
# collisions) is satisfied, but the model is never trained on sequences containing ``DUMMY_EOS`` so
# it will never emit this token greedily. Rolling generation is bounded solely by ``max_new_tokens``.
DUMMY_EOS = 18
VOCAB_SIZE = 20
MAX_SEQ_LEN = 16


def _start_token_to_name(token: int) -> str | None:
    for name, start in PROGRAM_STARTS.items():
        if token == start:
            return name
    return None


# ---------------------------------------------------------------------------
# Grammar FSM
# ---------------------------------------------------------------------------


class GrammarFSM:
    """Deterministic walker over the pattern grammar. Each ``step(token)`` returns either the new state name
    or ``None`` (meaning the transition is grammar-invalid).

    States:
      - ``"BETWEEN"`` — between programs (or at the very start). Valid next tokens: any
        program-start.
      - ``("IN", prog_name, pos)`` — inside program ``prog_name`` having just emitted position
        ``pos``. Valid next: either ``prog[pos + 1]`` if there is one, or (at the final position)
        ``SEP``.

    No terminal state: the training distribution never emits an EOS token (see module docstring),
    so the grammar is effectively infinite — any valid sequence of programs separated by ``SEP``
    can continue indefinitely.

    The FSM is used for two things in this file:

    1. Replaying a ground-truth prompt through the FSM to establish the initial state before we
       call the model (so the test can feed the model into the middle of a program and assert it
       completes that program correctly).
    2. Walking the model-generated continuation token-by-token and asserting every transition is
       valid.

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
    >>> fsm.step(8)  # would skip B[1]=7, invalid
    >>> fsm.state is None
    True
    """

    def __init__(self) -> None:
        self.state: str | tuple[str, str, int] | None = "BETWEEN"

    def step(self, token: int) -> str | tuple[str, str, int] | None:
        """Transition on ``token``. Returns new state, or ``None`` on invalid transition.

        Once the FSM enters the invalid state (``self.state is None``), further ``step`` calls
        remain in ``None``.
        """
        if self.state is None:
            return None
        if self.state == "BETWEEN":
            name = _start_token_to_name(token)
            if name is None:
                self.state = None
            else:
                self.state = ("IN", name, 0)
            return self.state
        # ("IN", name, pos)
        _, name, pos = self.state
        prog = PROGRAMS[name]
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
        """Walk a token sequence.

        Returns the index of the first invalid token, or ``len(tokens)``
        if every transition was valid.
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
    always ends with a ``SEP`` followed by nothing — i.e. the caller pads with ``PAD`` up to
    ``max_len``, and ``Model._forward``'s cross-entropy loss ignores those pad positions so the
    model never sees a "what comes after the final SEP" target. That keeps the trained grammar
    effectively non-terminating.

    >>> rng = random.Random(0)
    >>> seq = sample_sequence(rng)
    >>> GrammarFSM().walk(seq) == len(seq)
    True
    >>> seq[-1] == SEP
    True
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


def sample_batch(rng: random.Random, batch_size: int, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
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
        batch_codes = sample_batch(rng, _BATCH_SIZE)
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
    """For each prompt, greedy single-chunk generation must complete the in-progress program and then emit a
    grammar-valid transition (``SEP`` or ``EOS``)."""
    model = grammar_trained_model

    for prompt, expected_completion in _PROMPTS_AND_EXPECTED_COMPLETIONS:
        prompt_tensor = torch.tensor([prompt], dtype=torch.long)
        batch = mock_batch(prompt_tensor)

        generated = model.generate(batch, do_sample=False)  # single-chunk path
        produced = generated[0].tolist()

        # Strip any trailing PAD/EOS padding that appears after the model stopped emitting useful
        # tokens; we only need to check enough tokens to cover the expected completion plus one
        # more token (the transition out of the program).
        check_len = len(expected_completion) + 1
        assert len(produced) >= check_len, (
            f"Expected at least {check_len} generated tokens for prompt {prompt}, "
            f"got {len(produced)}: {produced}."
        )
        assert produced[: len(expected_completion)] == expected_completion, (
            f"Greedy completion of prompt {prompt} should be {expected_completion}; "
            f"model produced {produced[: len(expected_completion)]}."
        )

        # The token immediately after completing the program should be SEP — that's the only
        # valid grammar transition out of the final position of a program, and since the training
        # distribution never contains a terminating token, the model should always prefer it.
        transition_token = produced[len(expected_completion)]
        assert transition_token == SEP, (
            f"After completing prompt {prompt}, expected SEP={SEP} as the transition token; "
            f"got {transition_token}. Full output: {produced}."
        )


_ROLLING_BUDGET = 50


def test_trained_model_rolling_generation_preserves_grammar(grammar_trained_model: Model):
    """Rolling generation past the model's context window must stay grammar-valid.

    The trained model has ``max_position_embeddings=16`` and we generate
    ``max_new_tokens=_ROLLING_BUDGET`` from a short ``[A_start]`` prompt. That's strictly more than
    3x the context window, so the sliding-window loop has to cross several boundaries and emit a
    long tail of post-saturation single-token chunks. We then walk every generated token through
    the grammar FSM and assert each transition is valid. Because training never contained an EOS
    token (see module docstring) the model should emit exactly ``_ROLLING_BUDGET`` tokens;
    anything less means the rolling loop terminated early, which in this test setup is itself a
    regression.
    """
    model = grammar_trained_model

    # Seed with the start of program A.
    prompt = [PROGRAMS["A"][0]]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long)
    batch = mock_batch(prompt_tensor)

    # Spy on the inner HF call so we can assert the rolling loop genuinely iterated across many
    # sliding windows rather than degenerating to a single chunk call.
    real_generate = model.HF_model.generate
    from unittest.mock import MagicMock as _MagicMock

    spy = _MagicMock(wraps=real_generate)
    model.HF_model.generate = spy
    try:
        generated = model.generate(batch, do_sample=False, max_new_tokens=_ROLLING_BUDGET)
    finally:
        model.HF_model.generate = real_generate
    produced = generated[0].tolist()

    assert len(produced) == _ROLLING_BUDGET, (
        f"Rolling generation should emit exactly max_new_tokens={_ROLLING_BUDGET} tokens (training "
        f"data has no EOS so the loop shouldn't terminate early); got {len(produced)}. Full "
        f"output: {produced}."
    )

    # Inner call count: ``max_seq_len - 1 = 15`` per chunk before saturation, then 1 per chunk
    # after. With a 1-token prompt and a ``_ROLLING_BUDGET`` budget, the first chunk alone can
    # emit up to 15 tokens, then every subsequent chunk emits exactly 1. So for ``_ROLLING_BUDGET
    # = 50`` we expect at least ``1 + (50 - 15) = 36`` inner calls. We assert a loose ``>= 10`` so
    # the test is robust to harmless changes in window sizing but still fails if rolling ever
    # silently falls through to a single chunk.
    assert spy.call_count >= 10, (
        f"Rolling loop only made {spy.call_count} inner HF_model.generate call(s) for a budget "
        f"of {_ROLLING_BUDGET} tokens on a model with max_seq_len={MAX_SEQ_LEN} — expected many "
        f"rounds. The rolling path may not have been exercised."
    )

    # Replay the prompt through the FSM to establish the starting state for the continuation.
    fsm = GrammarFSM()
    for tok in prompt:
        fsm.step(tok)
    assert fsm.state == ("IN", "A", 0), (
        f"Expected FSM in ('IN', 'A', 0) after prompt {prompt}, got {fsm.state}."
    )

    # Walk the full continuation. Every transition must be grammar-valid.
    first_invalid = fsm.walk(produced)
    assert first_invalid == len(produced), (
        f"Rolling generation emitted a grammar-invalid token at position {first_invalid} "
        f"(token={produced[first_invalid]}). FSM state just before the violation: "
        f"{fsm.state}. Full output: {produced}."
    )

"""Pattern grammar for end-to-end autoregressive-generation tests.

Defines a tiny, finite-state grammar over integer tokens used by both the in-process test
(``tests/grammar/test_in_process.py`` — trains a ``Model`` directly on mock batches) and the
end-to-end CLI test (``tests/grammar/test_cli.py`` — drives the full ``MEICAR_process_data`` →
``MEICAR_pretrain`` → ``MEICAR_generate_trajectories`` pipeline). Both tests assert that the
trained model's generation stays grammar-valid; only the substrate differs.

The grammar has three named programs — ``A`` (4 tokens), ``B`` (3 tokens), ``C`` (9 tokens) —
emitted stochastically with a ``SEP`` token between them. Training sequences are packed full of
programs until no more fit, then padded — **no terminating EOS token**. That's intentional: if
we trained the model to emit EOS, the rolling-generation test would terminate early (the model
would emit EOS before the sliding window had a chance to cross a boundary) and the real
integration signal — "rolling generation stays grammar-valid across multiple sliding windows" —
would be smothered by early termination. Instead, we set the model's ``eos_token_id`` to an
out-of-grammar dummy token and let ``max_new_tokens`` control termination for the rolling test.

## What the training data looks like

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

Read left-to-right: program C (9, 10, ..., 17), then SEP. ``GrammarFSM`` accepts the full
sequence::

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
vacuous, this file also carries a second grammar ``ALT_PROGRAMS`` — same start tokens and same
token sets as ``PROGRAMS`` but with the internal order of each program permuted — and asserts
that ``PROGRAMS`` and ``ALT_PROGRAMS`` disagree on hand-built canonical sequences.

Concretely, program A is ``(2, 3, 4, 5)`` under ``PROGRAMS`` and ``(2, 4, 3, 5)`` under
``ALT_PROGRAMS``::

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
on what should follow A[0]=2).
"""

from __future__ import annotations

import random  # noqa: TC003  — referenced by module-level doctests (``random.Random(0)``)
from unittest.mock import Mock

import torch

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
    ``mock_batch`` inside the training loop would be noise.

    >>> rng = random.Random(0)
    >>> batch_codes = build_training_batch_codes(rng, batch_size=3)
    >>> batch_codes
    tensor([[ 9, 10, 11, 12, 13, 14, 15, 16, 17,  1,  0,  0,  0,  0,  0],
            [ 6,  7,  8,  1,  2,  3,  4,  5,  1,  6,  7,  8,  1,  0,  0],
            [ 9, 10, 11, 12, 13, 14, 15, 16, 17,  1,  2,  3,  4,  5,  1]])
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

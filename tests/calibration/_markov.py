"""A two-phase Markov chain whose exact next-token law is known in closed form.

The grammar fixture in :mod:`tests.grammar._grammar` establishes that generation is *grammatical*
— that the model's continuations stay inside the support of the training distribution. It cannot
say anything about whether generation is *calibrated*, i.e. whether sampled trajectories are
Monte Carlo draws from the model's own next-token distribution. Two properties of that fixture
make it structurally blind to miscalibrated sampling:

1. its vocabulary is 20 tokens, well under the ``top_k=50`` default that HF's ``GenerationConfig``
   silently installs, so the truncation is a no-op there; and
2. its conditional distributions are near-deterministic (each program token pins down the next),
   so almost all of the mass sits on a single token and truncating the tail changes nothing
   measurable.

This module is the complement: a source whose conditionals are *maximally spread* over a support
that is *comfortably wider than 50*, which is exactly the shape that makes top-k truncation
visible.

## The chain

Tokens are partitioned into two phases of :data:`N_TOKENS_PER_PHASE` tokens each, interleaved by
parity of their offset from :data:`FIRST_CONTENT_TOKEN`. The chain strictly alternates phase, and
within the next phase it is **uniform**::

    P(next = t' | current = t) = 1 / N_TOKENS_PER_PHASE   if phase(t') != phase(t)
                               = 0                        otherwise

So the model has something real to learn (the alternation), while the conditional it must
reproduce is a flat distribution over 100 tokens — twice the ``top_k=50`` cutoff. Under correct
ancestral sampling a long run of draws covers essentially the whole 100-token support; under the
top-50 default it is capped at 50 distinct tokens and the total-variation distance to the truth
is pinned at exactly ``(100 - 50) / 100 = 0.5``. That is a factor-of-four separation from the
sampling-noise floor, so the regression test needs no delicate thresholding.

Uniformity is a deliberate choice over a more "realistic" skewed conditional: it both maximizes
the excluded mass (a Zipf-like tail would put only ~25% of its mass outside the top 50) and makes
the ground truth a single number rather than a fitted curve.

## What the data looks like

    >>> import random
    >>> rng = random.Random(0)
    >>> seq = sample_sequence(rng, length=8)
    >>> seq
    [197, 108, 13, 68, 133, 126, 105, 78]

Read it as alternating phases — odd, even, odd, even, ... — with the token *within* each phase
drawn uniformly:

    >>> [phase_of(t) for t in seq]
    [1, 0, 1, 0, 1, 0, 1, 0]
    >>> all(t in PHASE_TOKENS[phase_of(t)] for t in seq)
    True

A training batch stacks several such sequences. Every sequence is exactly ``length`` tokens long,
so unlike the grammar fixture there is no padding to reason about:

    >>> rng = random.Random(0)
    >>> codes = build_training_batch_codes(rng, batch_size=3, length=6)
    >>> codes
    tensor([[197, 108,  13,  68, 133, 126],
            [ 79, 124,  93, 150,  57, 130],
            [ 74,  37, 194,  27, 160,  67]])
    >>> bool((codes != PAD).all())
    True

## The ground truth

:func:`true_next_token_distribution` returns the exact conditional as a vocabulary-wide vector,
which is what the calibration test compares empirical draw frequencies against:

    >>> p = true_next_token_distribution(seq[-1])
    >>> p.shape
    torch.Size([202])
    >>> round(float(p.sum()), 12)
    1.0
    >>> int((p > 0).sum()) == N_TOKENS_PER_PHASE
    True
    >>> sorted(set(p[p > 0].tolist()))
    [0.01]

Its support is the *opposite* phase from the conditioning token, and nothing else:

    >>> set((p > 0).nonzero().flatten().tolist()) == set(PHASE_TOKENS[1 - phase_of(seq[-1])])
    True

:func:`total_variation` scores a candidate distribution against it. Identical distributions score
0; a distribution supported on the wrong phase scores 1:

    >>> total_variation(p, p)
    0.0
    >>> q = true_next_token_distribution(PHASE_TOKENS[1 - phase_of(seq[-1])][0])
    >>> round(total_variation(p, q), 12)
    1.0

And — the case the regression test turns on — truncating the uniform conditional to its top 50
entries and renormalizing scores exactly 0.5:

    >>> import torch
    >>> truncated = torch.zeros_like(p)
    >>> kept = (p > 0).nonzero().flatten()[:50]
    >>> truncated[kept] = 1.0 / len(kept)
    >>> round(total_variation(p, truncated), 12)
    0.5
"""

from __future__ import annotations

import random  # noqa: TC003  — referenced by module-level doctests (``random.Random(0)``)
from unittest.mock import Mock

import torch

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD = 0

# Reserved out-of-distribution EOS. As in the grammar fixture, the chain never emits it, so
# generation is bounded purely by the token budget; it exists only so the model has a valid
# ``eos_token_id`` that differs from ``PAD``.
DUMMY_EOS = 1

FIRST_CONTENT_TOKEN = 2

# 100 tokens per phase — twice HF's ``top_k=50`` default. The factor of two is the whole point:
# it is what turns "the tail is being truncated" into an assertion with a wide margin rather
# than a threshold that has to be tuned.
N_TOKENS_PER_PHASE = 100

PHASE_TOKENS: tuple[tuple[int, ...], tuple[int, ...]] = (
    tuple(FIRST_CONTENT_TOKEN + 2 * i for i in range(N_TOKENS_PER_PHASE)),
    tuple(FIRST_CONTENT_TOKEN + 2 * i + 1 for i in range(N_TOKENS_PER_PHASE)),
)

VOCAB_SIZE = FIRST_CONTENT_TOKEN + 2 * N_TOKENS_PER_PHASE

MAX_SEQ_LEN = 16


def phase_of(token: int) -> int:
    """Return which of the two phases ``token`` belongs to.

    Examples:
        >>> phase_of(FIRST_CONTENT_TOKEN)
        0
        >>> phase_of(FIRST_CONTENT_TOKEN + 1)
        1
        >>> [phase_of(t) for t in PHASE_TOKENS[0][:3]]
        [0, 0, 0]
        >>> [phase_of(t) for t in PHASE_TOKENS[1][:3]]
        [1, 1, 1]
    """
    return (token - FIRST_CONTENT_TOKEN) % 2


def next_token_support(token: int) -> tuple[int, ...]:
    """Return the tokens reachable in one step from ``token`` (all equally likely).

    Examples:
        >>> next_token_support(PHASE_TOKENS[0][0]) == PHASE_TOKENS[1]
        True
        >>> next_token_support(PHASE_TOKENS[1][0]) == PHASE_TOKENS[0]
        True
        >>> len(next_token_support(PHASE_TOKENS[0][0]))
        100
    """
    return PHASE_TOKENS[1 - phase_of(token)]


def true_next_token_distribution(token: int) -> torch.Tensor:
    """Return the exact ``P(next | token)`` as a ``[VOCAB_SIZE]`` probability vector.

    This is the ground truth the calibration test scores sampled draws against — see the module docstring for
    worked examples.
    """
    p = torch.zeros(VOCAB_SIZE, dtype=torch.float64)
    support = next_token_support(token)
    p[list(support)] = 1.0 / len(support)
    return p


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two distributions over the same support.

    ``0`` for identical distributions, ``1`` for disjoint support. See the module docstring for
    the three cases that matter here (identical / disjoint / top-50-truncated).
    """
    return float(0.5 * (p.double() - q.double()).abs().sum())


# ---------------------------------------------------------------------------
# Sampling from the chain
# ---------------------------------------------------------------------------


def sample_sequence(rng: random.Random, length: int = MAX_SEQ_LEN) -> list[int]:
    """Draw one length-``length`` realization of the chain, starting from a uniform random token.

    Examples:
        >>> import random
        >>> seq = sample_sequence(random.Random(1), length=6)
        >>> seq
        [146, 197, 18, 67, 32, 129]

        Every consecutive pair is a legal transition — i.e. each token lies in the support
        implied by its predecessor:

        >>> all(b in next_token_support(a) for a, b in zip(seq[:-1], seq[1:]))
        True
    """
    token = rng.choice(PHASE_TOKENS[rng.randrange(2)])
    out = [token]
    while len(out) < length:
        token = rng.choice(next_token_support(token))
        out.append(token)
    return out


def build_training_batch_codes(
    rng: random.Random, batch_size: int, length: int = MAX_SEQ_LEN
) -> torch.Tensor:
    """Build a ``[batch_size, length]`` token-code tensor of independent chain realizations.

    Every row is exactly ``length`` tokens, so there is no padding and no ``ignore_index``
    interaction to reason about — unlike :func:`tests.grammar._grammar.build_training_batch_codes`,
    whose rows are variable-length programs.

    Examples:
        >>> import random
        >>> build_training_batch_codes(random.Random(1), batch_size=2, length=5)
        tensor([[146, 197,  18,  67,  32],
                [197, 116, 123, 168,  99]])
    """
    return torch.tensor(
        [sample_sequence(rng, length) for _ in range(batch_size)],
        dtype=torch.long,
    )


def mock_batch(code: torch.Tensor) -> Mock:
    """Build a minimal ``MEDSTorchBatch``-shaped mock that ``Model`` can consume."""
    return Mock(code=code, PAD_INDEX=PAD, mode="SM")

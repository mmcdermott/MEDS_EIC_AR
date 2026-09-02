"""Regression tests: sampled trajectories must be calibrated draws, not top-k truncated ones.

``Model._generate_chunk`` builds the ``GenerationConfig`` that governs every trajectory this repo
produces, and untruncated ancestral sampling is the only sampling mode the model supports. The
hazard is that ``transformers``' ``GenerationConfig`` defaults ``top_k`` to **50**, which installs
a ``TopKLogitsWarper`` unconditionally. Leaving that field unset — while setting ``do_sample``,
``num_beams`` and ``temperature`` right next to it — silently reduces every sampled trajectory to
a draw from the renormalized top-50 head of the vocabulary: a mode-seeking bias, invisible at the
call site, that invalidates every downstream consumer treating trajectories as Monte Carlo
samples from the model.

Three tests, in increasing cost and decreasing directness:

1. :func:`test_generate_chunk_requests_untruncated_sampling` — inspects the ``GenerationConfig``
   that reaches the backend. Sub-second, no training, no statistics; this is the test that will
   actually name the problem if the explicit ``top_k=0`` is ever dropped.
2. :func:`test_sampling_is_calibrated_against_the_true_markov_law` — the behavioral test: train a
   tiny model on a Markov chain with a known 100-token uniform conditional (see
   :mod:`tests.calibration._markov`), draw from it, and check the empirical law against the truth.
3. :func:`test_probe_detects_top_k_truncation` — the negative control for (2), which re-runs the
   same probe against a fault-injected backend that reinstates ``top_k=50`` and asserts the probe
   *fails* there. Without it, (2) could silently degrade into a test that passes for the wrong
   reason.
"""

from __future__ import annotations

import random

import pytest
import torch
from transformers import GenerationConfig

from MEDS_EIC_AR.model.model import Model
from tests.calibration._markov import (
    DUMMY_EOS,
    MAX_SEQ_LEN,
    N_TOKENS_PER_PHASE,
    VOCAB_SIZE,
    build_training_batch_codes,
    mock_batch,
    next_token_support,
    sample_sequence,
    total_variation,
    true_next_token_distribution,
)

# ---------------------------------------------------------------------------
# (1) Direct config assertion — no training, no statistics
# ---------------------------------------------------------------------------


class _ConfigCapturingBackend:
    """A ``GenerationBackend`` that records the config it is handed and emits padding.

    Returning ``pad_token_id`` at every position satisfies the backend contract's post-EOS
    invariant trivially, which is all the caller needs — this backend exists to inspect the
    ``GenerationConfig``, not to generate anything.
    """

    def __init__(self):
        self.configs: list[GenerationConfig] = []

    def generate_chunk(self, input_ids, *, attention_mask, generation_config, **kwargs):
        self.configs.append(generation_config)
        return torch.full(
            (input_ids.shape[0], generation_config.max_new_tokens),
            generation_config.pad_token_id,
            dtype=torch.long,
        )


@pytest.mark.parametrize("rolling", [False, True], ids=["single_chunk", "rolling"])
def test_generate_chunk_requests_untruncated_sampling(rolling: bool):
    """Both generation paths must ask the backend for untruncated ancestral sampling.

    ``top_k == 0`` (not ``None``, not left unset) is the load-bearing assertion: HF adds a
    ``TopKLogitsWarper`` whenever ``top_k`` is neither ``None`` nor ``0``, and
    ``GenerationConfig()`` defaults it to 50. The other three fields are asserted alongside it so
    the *whole* sampling law is pinned at the call site rather than half-inherited from
    ``transformers``.

    Parametrized over both paths because they compute their budgets differently and a future
    refactor could plausibly fix one and not the other.
    """
    model = Model(
        {
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 8,
            "vocab_size": 32,
            "eos_token_id": 3,
        },
        precision="32-true",
    )
    backend = _ConfigCapturingBackend()
    model.set_backend(backend)

    batch = mock_batch(torch.tensor([[4, 5, 6, 7]], dtype=torch.long))
    if rolling:
        model.generate(batch, do_sample=True, max_new_tokens=6)
    else:
        model.generate(batch, do_sample=True)

    assert backend.configs, "The backend was never called; the test cannot have proved anything."
    for cfg in backend.configs:
        assert cfg.top_k == 0, (
            f"GenerationConfig.top_k is {cfg.top_k!r}, not 0. HF installs a TopKLogitsWarper for "
            "any value that is neither None nor 0 (its class default is 50), which silently turns "
            "sampling into a draw from the truncated top-k head instead of from the model's own "
            "next-token distribution."
        )
        assert cfg.top_p == 1.0, f"GenerationConfig.top_p is {cfg.top_p!r}, not 1.0 (no nucleus truncation)."
        assert cfg.temperature == 1.0, f"GenerationConfig.temperature is {cfg.temperature!r}, not 1.0."
        assert cfg.num_beams == 1, f"GenerationConfig.num_beams is {cfg.num_beams!r}, not 1."


def test_generation_config_default_top_k_is_still_the_hazard():
    """Guard the *premise* of this module: HF's default ``top_k`` really is truncating.

    If a future ``transformers`` release changed ``GenerationConfig().top_k`` to ``0`` or ``None``,
    the tests here would keep passing while no longer testing anything, and the explicit
    ``top_k=0`` in ``_generate_chunk`` would read as dead code to the next person through. This
    fails loudly in that case so the situation gets re-evaluated rather than quietly forgotten.
    """
    assert GenerationConfig().top_k == 50, (
        "transformers' GenerationConfig no longer defaults top_k to 50. The explicit top_k=0 in "
        "Model._generate_chunk may no longer be load-bearing — re-read the HF defaults and update "
        "this module's rationale accordingly."
    )


# ---------------------------------------------------------------------------
# (2) + (3) Behavioral calibration probe against a trained model
# ---------------------------------------------------------------------------

_NUM_TRAIN_STEPS = 200
_BATCH_SIZE = 32
_LEARNING_RATE = 3e-3

# Draws per probe. 2000 is far more than enough to cover a 100-token uniform support (the chance
# of missing any given token is (99/100)**2000 ~ 1e-9) while keeping the probe under a second.
_N_DRAWS = 2000

# Thresholds. The two regimes are separated by roughly a factor of four on every statistic, so
# these sit near the midpoint rather than being tuned:
#
#   statistic          | truncated to top 50 | untruncated  | threshold
#   -------------------|---------------------|--------------|-----------
#   distinct tokens    | <= 50 (hard cap)    | ~115-125     | >= 70
#   TV vs. true law    | exactly 0.50        | ~0.11-0.13   | <= 0.25
#   mass in support    | ~1.00               | ~0.99        | >= 0.95
#
# ``distinct`` is a hard cap under truncation, not a tendency: the probe conditions every draw on
# the *same* prompt, so the model's logits — and hence the retained top-50 set — are identical
# across draws.
_MIN_DISTINCT_TOKENS = 70
_MAX_TV_TO_TRUTH = 0.25
_MIN_MASS_IN_SUPPORT = 0.95


@pytest.fixture(scope="module")
def markov_trained_model() -> Model:
    """Train a tiny ``Model`` on the two-phase Markov chain (a few seconds on CPU).

    Module-scoped so the calibration test and its negative control share one model — they must, in fact, since
    the control's whole job is to show that the *same* model probed the *same* way fails once truncation is
    reintroduced.
    """
    torch.manual_seed(0)
    rng = random.Random(0)

    model = Model(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": MAX_SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "eos_token_id": DUMMY_EOS,
        },
        precision="32-true",
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    for _ in range(_NUM_TRAIN_STEPS):
        optimizer.zero_grad()
        loss, _ = model(mock_batch(build_training_batch_codes(rng, _BATCH_SIZE)))
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _probe(model: Model) -> dict[str, float]:
    """Draw ``_N_DRAWS`` next-tokens from one fixed prompt and score them against the true law.

    The prompt is ``MAX_SEQ_LEN - 1`` tokens long, so the single-chunk path's budget
    (``max_seq_len - input_len``) is exactly one token — every row's output *is* the quantity to
    score, with no rolling-loop or EOS bookkeeping in the way. Every row of the batch carries the
    identical prompt, so all ``_N_DRAWS`` draws are i.i.d. samples from one conditional and their
    frequencies estimate that single distribution.
    """
    prompt = sample_sequence(random.Random(1), length=MAX_SEQ_LEN - 1)
    support = next_token_support(prompt[-1])
    truth = true_next_token_distribution(prompt[-1])

    codes = torch.tensor([prompt], dtype=torch.long).expand(_N_DRAWS, -1).contiguous()

    torch.manual_seed(0)
    with torch.no_grad():
        generated = model.generate(mock_batch(codes), do_sample=True)

    assert generated.shape == (_N_DRAWS, 1), (
        f"Probe expected exactly one generated token per row, got shape {tuple(generated.shape)}. "
        "The prompt-length / max_seq_len relationship this probe relies on has changed."
    )

    counts = torch.bincount(generated[:, 0], minlength=VOCAB_SIZE).double()
    empirical = counts / counts.sum()

    return {
        "distinct": int((counts > 0).sum()),
        "tv_to_truth": total_variation(empirical, truth),
        "mass_in_support": float(empirical[list(support)].sum()),
    }


def test_sampling_is_calibrated_against_the_true_markov_law(markov_trained_model: Model):
    """Draws must look like the model's true 100-token uniform conditional, not its top-50 head.

    ``mass_in_support`` is the "did the model actually learn the chain?" guard — it stops the
    other two assertions from passing vacuously on a model that learned nothing and is emitting
    near-uniform noise over the whole vocabulary. (TV alone already constrains that case, but
    stating the support check separately makes the failure diagnosable.) Note that *truncated*
    sampling scores better on this statistic — concentrating on the head is exactly what it does —
    which is precisely why the grammaticality tests elsewhere in this suite cannot detect
    miscalibration and this module has to exist.
    """
    stats = _probe(markov_trained_model)

    assert stats["mass_in_support"] >= _MIN_MASS_IN_SUPPORT, (
        f"Only {stats['mass_in_support']:.3f} of sampled mass landed in the true "
        f"{N_TOKENS_PER_PHASE}-token support, below {_MIN_MASS_IN_SUPPORT}. The fixture model has "
        "not learned the chain, so the calibration assertions below would be meaningless. Raise "
        "_NUM_TRAIN_STEPS."
    )
    assert stats["distinct"] >= _MIN_DISTINCT_TOKENS, (
        f"Only {stats['distinct']} distinct tokens appeared across {_N_DRAWS} draws from a "
        f"conditional that is uniform over {N_TOKENS_PER_PHASE}. A value at or below 50 means "
        "sampling is being truncated to a top-50 head; check that Model._generate_chunk still "
        "sets top_k=0."
    )
    assert stats["tv_to_truth"] <= _MAX_TV_TO_TRUTH, (
        f"Total-variation distance between the empirical draws and the true conditional is "
        f"{stats['tv_to_truth']:.3f}, above {_MAX_TV_TO_TRUTH}. A value near 0.5 is the signature "
        f"of top-50 truncation of a uniform-over-{N_TOKENS_PER_PHASE} law."
    )


class _TruncatingBackend:
    """Fault injection: a backend that reinstates ``top_k=50`` before delegating to the real one.

    Truncated sampling is not a supported mode, so there is no production knob to flip for the
    negative control below. Rewriting the config inside a wrapper backend reproduces exactly what
    an unset ``top_k`` would have done — HF resolves the field the same way whether it arrived as
    a class default or was written in — while leaving the prompt, the weights, the seed, and every
    other part of the probe identical.
    """

    def __init__(self, inner):
        self.inner = inner

    def generate_chunk(self, input_ids, *, attention_mask, generation_config, **kwargs):
        generation_config.top_k = 50
        return self.inner.generate_chunk(
            input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs
        )


def test_probe_detects_top_k_truncation(markov_trained_model: Model):
    """Negative control: reinstate top-50 truncation and confirm the probe rejects it.

    Without this, a probe that quietly stopped measuring anything — a prompt that pinned the
    conditional to a single token, a ``bincount`` over the wrong axis — would leave the
    calibration test green forever.
    """
    model = markov_trained_model
    original_backend = model.backend
    model.set_backend(_TruncatingBackend(original_backend))
    try:
        stats = _probe(model)
    finally:
        model.set_backend(original_backend)

    assert stats["distinct"] <= 50, (
        f"Truncated sampling yielded {stats['distinct']} distinct tokens, but a top-50 head caps a "
        "fixed-prompt conditional at 50 by construction. The probe is not measuring what it claims "
        "to."
    )
    assert stats["distinct"] < _MIN_DISTINCT_TOKENS
    assert stats["tv_to_truth"] > _MAX_TV_TO_TRUTH, (
        f"Truncated sampling gave TV {stats['tv_to_truth']:.3f}, which the calibration test would "
        f"have accepted (threshold {_MAX_TV_TO_TRUTH}). The probe has lost its sensitivity."
    )
    # And the statistic that truncation *improves*, spelled out: a grammaticality-style check
    # would have rated the broken configuration at least as highly as the correct one.
    assert stats["mass_in_support"] >= _MIN_MASS_IN_SUPPORT

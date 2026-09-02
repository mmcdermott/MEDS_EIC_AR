"""Sampling must draw from the model's own next-token distribution, untruncated.

``Model._generate_chunk`` builds the ``GenerationConfig`` that governs every trajectory this repo
produces. The hazard it has to defend against is that ``transformers``' ``GenerationConfig``
defaults ``top_k`` to **50**, which installs a ``TopKLogitsWarper`` unconditionally. Leaving that
field unset — while setting ``do_sample``, ``num_beams`` and ``temperature`` right beside it —
silently reduces every sampled trajectory to a draw from the renormalized top-50 head of the
vocabulary: a mode-seeking bias, invisible at the call site, that invalidates every downstream
consumer treating trajectories as Monte Carlo samples from the model.

The property under test is therefore a direct one — *the tokens generation emits are distributed
as the model's own softmax over the next token* — and it can be checked without training anything.
Take a model over a vocabulary comfortably larger than 50, read its next-token distribution ``P``
straight off a forward pass, draw many single-token continuations from the same prompt through
``Model.generate``, and compare the empirical distribution ``P̂`` against ``P``. No learned
generative process needs to sit in between: what makes a sample calibrated is agreement with the
model, and the model's distribution is right there to be read.

Three tests:

1. :func:`test_generate_chunk_requests_untruncated_sampling` — inspects the ``GenerationConfig``
   that reaches the backend, on both generation paths. Sub-second; this is the test that will name
   the problem if the explicit ``top_k=0`` is ever dropped.
2. :func:`test_sampled_tokens_match_the_models_own_distribution` — the behavioral check described
   above.
3. :func:`test_probe_detects_top_k_truncation` — the negative control for (2), which re-runs the
   same probe against a fault-injected backend that reinstates ``top_k=50`` and asserts the probe
   rejects it. Nothing here asserts anything about what HF's defaults happen to be: the property
   under test is our own sampling law, and this control keeps the probe honest without reaching
   for a third-party default to do it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

from MEDS_EIC_AR.model.model import Model

if TYPE_CHECKING:
    from transformers import GenerationConfig

PAD = 0

#: Vocabulary size for the probe model. Ten times the ``top_k=50`` cutoff, so truncation removes
#: most of the distribution rather than a sliver of tail.
VOCAB_SIZE = 512

MAX_SEQ_LEN = 8

#: Multiplier applied to the output head's weights after initialization. A freshly initialized
#: model's next-token distribution is nearly uniform (entropy within 1% of ``log(VOCAB_SIZE)``),
#: which would make the calibration assertion satisfiable by a sampler that ignored the model
#: entirely and drew uniformly. Scaling the head sharpens the distribution into one with a real
#: head and a real tail — far from uniform, but still with ~47% of its mass outside the top 50, so
#: truncation stays plainly visible. :func:`test_sampled_tokens_match_the_models_own_distribution`
#: asserts the "far from uniform" half of that as a premise.
_LM_HEAD_SHARPENING = 12.0

#: Draws per probe. At this vocabulary the multinomial sampling noise floor on the
#: total-variation distance is ~0.07; see the threshold table below.
_N_DRAWS = 10_000

# Thresholds. Measured across six model seeds x two generation seeds, correct vs. truncated:
#
#   statistic                        | truncated to top 50 | untruncated   | threshold
#   ---------------------------------|---------------------|---------------|-----------
#   TV(P, P-hat)                     | 0.47 - 0.52         | 0.069 - 0.078 | <= 0.20
#   distinct tokens over 10k draws   | 50 (a hard cap)     | 472 - 489     | >= 200
#   TV(P, uniform)  [premise]        | 0.47 - 0.50         | same          | >= 0.25
#
# The first two sit at roughly the midpoint of a six-fold gap, so neither needed tuning. Note that
# ``distinct <= 50`` under truncation is a hard cap rather than a tendency: every draw conditions on
# the same prompt, so the model's logits — and hence the retained top-50 set — are identical across
# draws.
_MAX_TV_TO_MODEL = 0.20
_MIN_DISTINCT_TOKENS = 200
_MIN_TV_MODEL_VS_UNIFORM = 0.25


def _mock_batch(code: torch.Tensor) -> Mock:
    """Build a minimal ``MEDSTorchBatch``-shaped mock that ``Model`` can consume."""
    return Mock(code=code, PAD_INDEX=PAD, mode="SM")


def _probe_model(seed: int = 0) -> Model:
    """A small randomly-initialized model with a deliberately sharpened output head."""
    torch.manual_seed(seed)
    model = Model(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "max_position_embeddings": MAX_SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "eos_token_id": 1,
        },
        precision="32-true",
    )
    with torch.no_grad():
        model.HF_model.lm_head.weight.mul_(_LM_HEAD_SHARPENING)
    model.eval()
    return model


#: The prompt every probe conditions on. ``MAX_SEQ_LEN - 1`` tokens long, so the single-chunk
#: path's budget (``max_seq_len - input_len``) is exactly one token and each row's output *is* the
#: draw to be scored — no rolling loop or EOS bookkeeping in between.
_PROMPT = torch.arange(2, 2 + MAX_SEQ_LEN - 1, dtype=torch.long).unsqueeze(0)


def _model_next_token_distribution(model: Model) -> torch.Tensor:
    """``P``: the model's own softmax over the token following :data:`_PROMPT`.

    Routed through ``Model._hf_inputs`` rather than hand-built so the forward pass sees exactly the
    ids and attention mask that ``Model.generate`` would construct from the same batch. With the
    sampling law this repo pins (``temperature=1.0``, no top-k, no top-p) HF applies no logits
    warper at all, so this softmax is precisely the distribution ``generate`` should be sampling.
    """
    with torch.no_grad():
        logits = model.HF_model(**model._hf_inputs(_mock_batch(_PROMPT))).logits[0, -1]
    return torch.softmax(logits.double(), dim=-1)


def _empirical_next_token_distribution(model: Model, seed: int = 0) -> tuple[torch.Tensor, int]:
    """``P̂``: the distribution of ``_N_DRAWS`` next-tokens drawn through ``Model.generate``.

    Every row of the batch carries the identical prompt, so the draws are i.i.d. samples from one conditional
    and their frequencies estimate that single distribution. Returns the distribution and the number of
    distinct tokens observed.
    """
    codes = _PROMPT.expand(_N_DRAWS, -1).contiguous()

    torch.manual_seed(seed)
    with torch.no_grad():
        generated = model.generate(_mock_batch(codes), do_sample=True)

    assert generated.shape == (_N_DRAWS, 1), (
        f"Probe expected exactly one generated token per row, got shape {tuple(generated.shape)}. "
        "The prompt-length / max_seq_len relationship this probe relies on has changed."
    )

    counts = torch.bincount(generated[:, 0], minlength=VOCAB_SIZE).double()
    return counts / counts.sum(), int((counts > 0).sum())


def _total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance: ``0`` for identical distributions, ``1`` for disjoint support."""
    return float(0.5 * (p.double() - q.double()).abs().sum())


# ---------------------------------------------------------------------------
# Config assertions — no sampling, no statistics
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
    ``transformers``. In particular this is what holds ``temperature``, which the behavioral test
    below is deliberately insensitive to.

    Parametrized over both paths because they compute their budgets differently and a future
    refactor could plausibly fix one and not the other.
    """
    model = _probe_model()
    backend = _ConfigCapturingBackend()
    model.set_backend(backend)

    batch = _mock_batch(_PROMPT)
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


# ---------------------------------------------------------------------------
# Behavioral check + negative control
# ---------------------------------------------------------------------------


def test_sampled_tokens_match_the_models_own_distribution():
    """Draws from ``generate`` must be distributed as the model's own softmax over the next token.

    The first assertion is a premise about the fixture, not about the code under test: if ``P`` were
    (near-)uniform, ``P̂ ≈ P`` would also hold for a sampler that ignored the model entirely, and the
    test would be vacuous. The sharpened output head is what buys a ``P`` far enough from uniform
    for agreement with it to mean something.
    """
    model = _probe_model()
    p_model = _model_next_token_distribution(model)
    p_sampled, distinct = _empirical_next_token_distribution(model)

    uniform = torch.full_like(p_model, 1.0 / VOCAB_SIZE)
    tv_vs_uniform = _total_variation(p_model, uniform)
    assert tv_vs_uniform >= _MIN_TV_MODEL_VS_UNIFORM, (
        f"The probe model's next-token distribution is {tv_vs_uniform:.3f} away from uniform, below "
        f"{_MIN_TV_MODEL_VS_UNIFORM}. Agreement with a near-uniform P would prove nothing, since a "
        "sampler that ignored the model would agree too. Raise _LM_HEAD_SHARPENING."
    )

    tv = _total_variation(p_model, p_sampled)
    assert tv <= _MAX_TV_TO_MODEL, (
        f"Total-variation distance between the sampled tokens and the model's own next-token "
        f"distribution is {tv:.3f}, above {_MAX_TV_TO_MODEL} (multinomial noise alone accounts for "
        f"~0.07 at this vocabulary and draw count). Generation is not sampling from the "
        "distribution it should be — most likely a logits warper is truncating it."
    )
    assert distinct >= _MIN_DISTINCT_TOKENS, (
        f"Only {distinct} distinct tokens appeared across {_N_DRAWS} draws from a "
        f"{VOCAB_SIZE}-token distribution. A value at or below 50 means sampling is being truncated "
        "to a top-50 head; check that Model._generate_chunk still sets top_k=0."
    )


class _TruncatingBackend:
    """Fault injection: a backend that reinstates ``top_k=50`` before delegating to the real one.

    Truncated sampling is not a supported mode, so there is no production knob to flip for the
    negative control below. Rewriting the config inside a wrapper backend reproduces exactly what
    an unset ``top_k`` would have done — HF resolves the field the same way whether it arrived as a
    class default or was written in — while leaving the prompt, the weights, the seed and every
    other part of the probe identical.
    """

    def __init__(self, inner):
        self.inner = inner

    def generate_chunk(self, input_ids, *, attention_mask, generation_config, **kwargs):
        generation_config.top_k = 50
        return self.inner.generate_chunk(
            input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs
        )


def test_probe_detects_top_k_truncation():
    """Negative control: reinstate top-50 truncation and confirm the probe rejects it.

    Without this, a probe that quietly stopped measuring anything — a prompt that pinned the
    conditional to a single token, a ``bincount`` over the wrong axis — would leave the calibration
    test green forever.
    """
    model = _probe_model()
    p_model = _model_next_token_distribution(model)
    model.set_backend(_TruncatingBackend(model.backend))
    p_sampled, distinct = _empirical_next_token_distribution(model)

    assert distinct <= 50, (
        f"Truncated sampling yielded {distinct} distinct tokens, but a top-50 head caps a "
        "fixed-prompt conditional at 50 by construction. The probe is not measuring what it claims "
        "to."
    )
    assert distinct < _MIN_DISTINCT_TOKENS
    tv = _total_variation(p_model, p_sampled)
    assert tv > _MAX_TV_TO_MODEL, (
        f"Truncated sampling gave TV {tv:.3f}, which the calibration test would have accepted "
        f"(threshold {_MAX_TV_TO_MODEL}). The probe has lost its sensitivity."
    )

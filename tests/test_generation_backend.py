"""Unit tests for the generation-backend abstraction (issue #88, step 1 of 2).

These tests only exercise the :class:`GenerationBackend` protocol contract and the
:class:`HFBackend` adapter. They deliberately *do not* drive the full CLI pipeline — that's
covered by ``test_generate_trajectories.py`` and the end-to-end grammar test at
``tests/grammar/test_cli.py``. The goal here is to lock in the contract that lets PR 2
(SGLang implementation) slot in without touching the rolling loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from MEDS_EIC_AR.model.backends import GenerationBackend, HFBackend
from MEDS_EIC_AR.model.model import Model

if TYPE_CHECKING:
    from transformers import GenerationConfig


def _tiny_model() -> Model:
    return Model(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 4,
            "max_position_embeddings": 10,
            "vocab_size": 40,
        },
        precision="32-true",
    )


def test_default_backend_is_hf():
    """``Model`` ships with an :class:`HFBackend` that wraps its own ``HF_model``.

    If this breaks, either the default was forgotten in the constructor or someone reassigned the backend
    attribute during init.
    """
    model = _tiny_model()
    assert isinstance(model.backend, HFBackend)
    assert isinstance(model.backend, GenerationBackend)  # protocol check
    assert model.backend.hf_model is model.HF_model


def test_hf_backend_slices_prompt_off_output():
    """HFBackend must return *only* the newly generated tokens, not prompt + new.

    This is the invariant the rolling loop depends on — every accumulator in ``_rolling_generate``
    assumes ``generate_chunk`` returns ``[B, new_len]``, not ``[B, L_in + new_len]``. We prove
    the contract by asserting byte-equality with ``HF_model.generate(...)`` sliced by the prompt
    length: the adapter must be a no-op on top of the underlying call, and specifically must not
    introduce its own sampling noise or truncation. Upper-bounding on ``max_new_tokens`` keeps
    the test robust if a random-init tiny model happens to emit EOS early.
    """
    from transformers import GenerationConfig

    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    cfg = GenerationConfig(
        max_new_tokens=2,
        do_sample=False,
        num_beams=1,
        pad_token_id=0,
        eos_token_id=model.HF_model.config.eos_token_id,
    )
    expected = model.HF_model.generate(
        input_ids=input_ids, attention_mask=attention_mask, generation_config=cfg
    )[:, input_ids.shape[1] :]
    out = model.backend.generate_chunk(input_ids, attention_mask=attention_mask, generation_config=cfg)

    assert out.shape[0] == input_ids.shape[0]
    assert out.shape[1] <= cfg.max_new_tokens
    assert torch.equal(out, expected)


class _RecordingBackend:
    """A backend that records every call and returns a deterministic fixed tensor.

    Used to prove that swapping the backend via :meth:`Model.set_backend` actually redirects the
    single inner call in ``_generate_chunk`` — i.e. no code path bypasses the backend and reaches
    ``HF_model.generate`` directly.
    """

    def __init__(self, new_tokens: torch.Tensor):
        self._new_tokens = new_tokens
        self.calls: list[dict] = []

    def generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        generation_config: GenerationConfig,
        **kwargs,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "input_ids_shape": tuple(input_ids.shape),
                "max_new_tokens": generation_config.max_new_tokens,
                "do_sample": generation_config.do_sample,
            }
        )
        batch_size = input_ids.shape[0]
        new_len = generation_config.max_new_tokens
        return self._new_tokens[:batch_size, :new_len].to(input_ids.device)


def test_set_backend_redirects_all_generate_calls():
    """Every ``Model._generate_chunk`` path must route through ``self._backend``.

    If a future refactor reintroduces a direct ``self.HF_model.generate`` call anywhere, this
    test breaks — the recording backend would not see the call.
    """
    model = _tiny_model()
    # Fake "new tokens" the backend will return: 5 tokens of a fixed id so we can assert the
    # caller gets exactly what the backend produced, not what the underlying HF model would have.
    new_tokens = torch.full((2, 5), fill_value=7, dtype=torch.long)
    recorder = _RecordingBackend(new_tokens)
    model.set_backend(recorder)
    assert model.backend is recorder

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model._generate_chunk(
        input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        max_new_tokens=3,
        pad_id=0,
        do_sample=False,
    )

    # The recorder wins — output is whatever the fake returned.
    assert torch.equal(out, torch.full((2, 3), fill_value=7, dtype=torch.long))
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["input_ids_shape"] == (2, 3)
    assert recorder.calls[0]["max_new_tokens"] == 3
    assert recorder.calls[0]["do_sample"] is False


def test_backend_protocol_structural_check():
    """A plain-Python class implementing ``generate_chunk`` satisfies :class:`GenerationBackend`.

    ``GenerationBackend`` is a ``@runtime_checkable`` Protocol so the check is truly structural.
    This matters for PR 2 (SGLang): the SGLang adapter must not be forced to inherit any concrete
    base class we own.
    """
    recorder = _RecordingBackend(torch.zeros(2, 5, dtype=torch.long))
    assert isinstance(recorder, GenerationBackend)


# -- Kwarg-filter behavior --------------------------------------------------
# HFBackend's ``**kwargs`` handling has two cases:
#  (a) HF ``generate`` has a VAR_KEYWORD ``**kwargs`` — the common case — in which case we
#      forward all caller kwargs unchanged and let HF validate. This is what preserves
#      dynamic-kwarg generation controls like ``temperature`` / ``top_p`` / ``top_k`` that
#      flow through HF's catch-all rather than being explicit named parameters.
#  (b) HF ``generate`` has a fixed signature with no VAR_KEYWORD — a hypothetical future
#      transformers layout — in which case we fall back to name-filtering so unknown kwargs
#      don't reach HF and raise.
# The two tests below lock in that branching. The fake-model approach lets us verify
# behavior independently of whichever transformers signature shape is currently installed.


class _FakeVarKeywordModel:
    """A fake ``hf_model`` whose ``generate`` has a ``**kwargs`` VAR_KEYWORD parameter."""

    class _Cfg:
        eos_token_id = 0

    config = _Cfg()

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def generate(self, input_ids, attention_mask=None, generation_config=None, **kwargs):
        # Record what was forwarded so the test can assert on it.
        self.last_kwargs = dict(kwargs)
        # Return prompt + one dummy new token per row so the slice in HFBackend works.
        batch_size = input_ids.shape[0]
        new_tokens = torch.zeros((batch_size, 1), dtype=input_ids.dtype)
        return torch.cat([input_ids, new_tokens], dim=1)


class _FakeFixedSignatureModel:
    """A fake ``hf_model`` whose ``generate`` has *no* VAR_KEYWORD — only named params."""

    class _Cfg:
        eos_token_id = 0

    config = _Cfg()

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def generate(self, input_ids, attention_mask=None, generation_config=None, known_kwarg=None):
        self.last_kwargs = {"known_kwarg": known_kwarg}
        batch_size = input_ids.shape[0]
        new_tokens = torch.zeros((batch_size, 1), dtype=input_ids.dtype)
        return torch.cat([input_ids, new_tokens], dim=1)


def _trivial_generation_config():
    from transformers import GenerationConfig

    return GenerationConfig(max_new_tokens=1, do_sample=False, num_beams=1, pad_token_id=0)


def test_hf_backend_forwards_dynamic_kwargs_when_var_keyword_present():
    """VAR_KEYWORD case: ``temperature`` / ``top_p`` and other dynamic kwargs must reach HF unchanged.

    Regression guard for the earlier revision where the backend filtered ``**kwargs`` to names
    in ``inspect.signature(generate).parameters`` — which silently dropped VAR_KEYWORD-only
    options like ``temperature``. The current branch detects VAR_KEYWORD and skips filtering.
    """
    fake = _FakeVarKeywordModel()
    backend = HFBackend(fake)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    backend.generate_chunk(
        input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        generation_config=_trivial_generation_config(),
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        some_model_forward_kwarg=True,
    )
    # All four kwargs must have been forwarded — the filter is a no-op when VAR_KEYWORD is
    # present, so HF gets the chance to validate / consume them itself.
    assert fake.last_kwargs == {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "some_model_forward_kwarg": True,
    }


def test_hf_backend_strips_unknown_kwargs_when_no_var_keyword():
    """Fixed-signature case: kwargs not in the explicit parameter list get stripped.

    This is the fallback path for a hypothetical transformers where ``generate`` has no
    ``**kwargs`` VAR_KEYWORD. Without filtering, passing an unknown kwarg would raise
    ``TypeError``; with filtering, the unknown kwarg is silently dropped and known kwargs
    are preserved.
    """
    fake = _FakeFixedSignatureModel()
    backend = HFBackend(fake)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    # ``known_kwarg`` is a real param and should reach the model. ``some_sglang_thing`` is
    # unknown and must be stripped — without stripping, the call would TypeError.
    backend.generate_chunk(
        input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        generation_config=_trivial_generation_config(),
        known_kwarg="reaches_model",
        some_sglang_thing="stripped",
    )
    assert fake.last_kwargs == {"known_kwarg": "reaches_model"}

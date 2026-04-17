"""Unit tests for the generation-backend abstraction (issue #88, step 1 of 2).

These tests only exercise the :class:`GenerationBackend` protocol contract and the
:class:`HFBackend` adapter. They deliberately *do not* drive the full CLI pipeline — that's
covered by ``test_generate_trajectories.py`` and ``test_pattern_generation_cli.py``. The goal
here is to lock in the contract that lets PR 2 (SGLang implementation) slot in without
touching the rolling loop.
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
    assumes ``generate_chunk`` returns ``[B, new_len]``, not ``[B, L_in + new_len]``.
    """
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    from transformers import GenerationConfig

    cfg = GenerationConfig(
        max_new_tokens=2,
        do_sample=False,
        num_beams=1,
        pad_token_id=0,
        eos_token_id=model.HF_model.config.eos_token_id,
    )
    out = model.backend.generate_chunk(input_ids, attention_mask=attention_mask, generation_config=cfg)

    # New-tokens-only shape: batch preserved, length equals the per-call budget (no early EOS
    # expected from a randomly initialized tiny model in 2 steps, so this is exact).
    assert out.shape == (2, 2)


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

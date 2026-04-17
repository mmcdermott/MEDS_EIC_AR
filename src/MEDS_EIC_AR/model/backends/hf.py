"""HuggingFace :class:`~transformers.GenerationMixin` implementation of :class:`GenerationBackend`.

Wraps a ``GPTNeoXForCausalLM`` (or any other ``PreTrainedModel`` with a usable ``generate``
method) and forwards per-chunk calls straight through. This is the default backend; behavior is
byte-identical to the pre-abstraction path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import GenerationConfig, PreTrainedModel


class HFBackend:
    """Thin adapter around a HuggingFace ``PreTrainedModel.generate`` call.

    The adapter holds a reference to the model — it deliberately does **not** own the model, so
    parameter sharing with Lightning / checkpointing is unaffected. Every call slices off the
    prompt tokens before returning, matching the contract documented on
    :class:`GenerationBackend`.

    **On the ``**kwargs`` forwarding policy.** The :class:`GenerationBackend` protocol says
    implementations must "only forward options supported by the active engine and silently
    ignore or strip the rest." Here the "active engine" is HF ``generate``, whose
    :class:`~transformers.GenerationMixin` accepts a broad ``**kwargs`` set — anything the
    underlying model's ``forward`` accepts, plus all generation-control kwargs like
    ``logits_processor`` / ``stopping_criteria``. We therefore forward ``**kwargs`` unchanged
    rather than maintaining a brittle whitelist that would drift across transformers releases.
    This satisfies the protocol because HF's supported-kwarg set is a superset of what callers
    realistically pass through ``Model.generate``. The obligation to *strip* applies
    asymmetrically to backends whose engine accepts a narrower set than HF — an SGLang adapter
    (PR 2, issue #88) will need to drop HF-specific keys like ``logits_processor`` before
    invoking ``sgl.Engine.generate``.
    """

    def __init__(self, hf_model: PreTrainedModel):
        self.hf_model = hf_model

    def generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        generation_config: GenerationConfig,
        **kwargs,
    ) -> torch.Tensor:
        out = self.hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )
        return out[:, input_ids.size(1) :]

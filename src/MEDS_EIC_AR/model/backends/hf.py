"""HuggingFace :class:`~transformers.GenerationMixin` implementation of :class:`GenerationBackend`.

Wraps a ``GPTNeoXForCausalLM`` (or any other ``PreTrainedModel`` with a usable ``generate``
method) and forwards per-chunk calls straight through. This is the default backend; behavior is
byte-identical to the pre-abstraction path.
"""

from __future__ import annotations

import inspect
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
    ignore or strip the rest." HF's ``generate`` accepts a broad set (anything in
    :class:`~transformers.GenerationMixin`'s signature plus model-specific ``forward`` kwargs),
    so in practice almost any caller-provided key is either accepted or doesn't matter. But HF
    *does* raise on truly unknown kwargs, and future callers may pass backend-agnostic keys
    that are meaningful to other backends (e.g. an SGLang-specific throughput knob) but not to
    HF. To be robust, we filter ``**kwargs`` down to HF-accepted parameter names via
    :func:`inspect.signature` before forwarding. This keeps the forward path resilient to
    multi-backend calling code without us hand-maintaining a whitelist.
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
        if kwargs:
            supported = inspect.signature(self.hf_model.generate).parameters
            kwargs = {k: v for k, v in kwargs.items() if k in supported}
        out = self.hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )
        return out[:, input_ids.size(1) :]

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

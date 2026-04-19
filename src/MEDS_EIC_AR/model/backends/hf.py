"""HuggingFace :class:`~transformers.GenerationMixin` implementation of :class:`GenerationBackend`.

Wraps a ``LlamaForCausalLM`` (or any other ``PreTrainedModel`` with a usable ``generate``
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
    ignore or strip the rest." HF's ``generate`` accepts a broad set of kwargs — many of them
    (``temperature``, ``top_p``, ``top_k``, ``renormalize_logits``, model-forward kwargs, …)
    flow through a ``**kwargs`` VAR_KEYWORD parameter rather than appearing as named args in
    ``generate``'s signature, and the actual accepted set is validated dynamically inside
    ``generate`` against ``GenerationConfig`` fields and the underlying model's ``forward``.
    That makes a name-only signature filter strictly wrong — it would silently strip
    legitimate HF options like ``temperature`` that only exist as keys inside the VAR_KEYWORD.

    So the filter we apply depends on the signature shape:

    1. If ``generate`` has a VAR_KEYWORD parameter (the common case — it does on current
       transformers), we forward ``**kwargs`` unchanged. HF validates the keys itself and will
       raise a clear ``TypeError`` if a caller passes something truly unknown. Backends whose
       engine has a narrower accepted set (SGLang, vLLM) still need to strip keys themselves,
       but that asymmetry belongs in *their* adapters, not here.
    2. If no VAR_KEYWORD exists (a hypothetical future transformers where ``generate`` is a
       fixed-signature function), fall back to filtering to the explicit named parameters.
    """

    def __init__(self, hf_model: PreTrainedModel):
        self.hf_model = hf_model
        sig = inspect.signature(hf_model.generate)
        # If HF ``generate`` has a VAR_KEYWORD (``**kwargs``), don't filter — many valid keys
        # (``temperature``, ``top_p``, etc.) flow through it rather than being explicit
        # parameters, and the name-only filter would drop them. HF's own runtime validation
        # handles truly-unknown kwargs.
        self._has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        self._supported_generate_params = frozenset(sig.parameters)

    def generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        generation_config: GenerationConfig,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs and not self._has_var_keyword:
            kwargs = {k: v for k, v in kwargs.items() if k in self._supported_generate_params}
        out = self.hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )
        return out[:, input_ids.size(1) :]

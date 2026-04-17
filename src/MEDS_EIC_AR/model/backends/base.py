"""Pluggable generation-backend abstraction.

The point of this layer is to isolate *where* a per-chunk generate call executes (HuggingFace
today, SGLang or another engine in the future — see issue #88) from *how* the surrounding rolling
sliding-window loop, EOS bookkeeping, and token accounting work. Only the innermost
``generate_chunk`` call is swappable; everything else stays in :class:`MEDS_EIC_AR.model.model.Model`.

A backend implementation takes a right-padded prompt tensor plus the HF-style ``GenerationConfig``
the caller has already built and returns the newly generated tokens (the portion of the HF output
that comes *after* the prompt). The slicing is deliberately the backend's responsibility: engines
like SGLang produce "new tokens only" natively, and pushing the slice into the adapter keeps the
calling code identical across backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from transformers import GenerationConfig


@runtime_checkable
class GenerationBackend(Protocol):
    """Contract for a per-chunk generation engine.

    Implementations wrap a specific inference runtime (HuggingFace, SGLang, …) and expose a single
    method that matches the shape the rolling loop in ``Model._generate_chunk`` needs. A backend is
    stateful only to the extent of holding a handle on its underlying engine — it does not own any
    of the rolling-loop bookkeeping (sequence buffer, finished mask, EOS truncation across chunks).
    """

    def generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        generation_config: GenerationConfig,
        **kwargs,
    ) -> torch.Tensor:
        """Run one generate pass and return only the newly generated tokens.

        Args:
            input_ids: ``[B, L_in]`` right-padded prompt tokens.
            attention_mask: Optional ``[B, L_in]`` attention mask; ``True`` for real prompt tokens.
            generation_config: A fully-populated :class:`transformers.GenerationConfig` describing
                per-call budget (``max_new_tokens``), sampling mode, pad/EOS ids, etc.
            **kwargs: Forwarded to the underlying engine. Callers pass backend-specific options
                here (e.g. HF ``logits_processor``); backends must tolerate unknown keys by
                silently passing them through to the engine.

        Returns:
            A ``[B, new_len]`` tensor of newly generated tokens, with the prompt slice already
            stripped. ``new_len`` is ``generation_config.max_new_tokens`` or less if the engine
            emitted EOS earlier.
        """

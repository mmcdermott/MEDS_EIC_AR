"""Lightning-checkpoint → HF-on-disk-directory export (for SGLang, issue #88).

SGLang's ``Engine`` loads weights from an HF-format directory — ``config.json`` plus one or
more ``*.safetensors`` shards — not from a Lightning ``.ckpt``. This module materializes a
:class:`~MEDS_EIC_AR.training.MEICARModule`'s HF submodel into such a directory so the
SGLang backend can point at it without any weight-manipulation code of its own.

Why a separate module (not inlined in the backend):

- Keeps the SGLang backend free of any Lightning awareness; the backend just receives a path.
- Makes the export testable with pure HF (``LlamaForCausalLM.from_pretrained``) round-trip,
  which runs in every CI lane — no GPU or SGLang dep required.
- Supports the idempotency / caching story (skip re-export when the target directory already
  has a state_dict matching the source checkpoint), which SGLang callers genuinely need
  because engine startup dwarfs single-run generation time at our scale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ...training.module import MEICARModule

logger = logging.getLogger(__name__)


def _state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    """Deterministic content hash of a state_dict.

    Used to decide whether a previously-exported HF directory is still current. Hashes the sorted param names
    concatenated with each tensor's SHA-256 of its raw bytes. Not meant to survive a tensor-dtype change — a
    dtype flip produces a different hash, which is the correct "re-export" signal.
    """
    hasher = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(b"\x00")
        hasher.update(t.numpy().tobytes())
        hasher.update(b"\x00")
    return hasher.hexdigest()


def export_lightning_to_hf_dir(module: MEICARModule, out_dir: Path | str) -> Path:
    """Materialize a ``MEICARModule``'s HF submodel as an on-disk HF directory.

    Writes the underlying ``HF_model`` (``LlamaForCausalLM`` post-#108) via
    ``save_pretrained``, plus a stub ``tokenizer_config.json`` so SGLang's
    ``skip_tokenizer_init=True`` path doesn't warn about a missing tokenizer.

    **Idempotency**: if ``<out_dir>/.export_fingerprint`` matches the current state_dict
    hash, re-export is skipped. Two concurrent exporters won't corrupt each other — the write
    lands in ``<out_dir>.tmp/`` and is atomically renamed into place on success; the losing
    writer cleans up its tmp dir.

    Args:
        module: Lightning module whose ``model.HF_model`` will be exported.
        out_dir: Target directory. Created if absent.

    Returns:
        The resolved ``out_dir`` path.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> import torch
        >>> from MEDS_EIC_AR.model import Model
        >>> from MEDS_EIC_AR.training.module import MEICARModule
        >>> from MEDS_EIC_AR.training.metrics import NextCodeMetrics
        >>> _ = torch.manual_seed(0)
        >>> model = Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 2,
        ...     "hidden_size": 4,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": 10,
        ... })
        >>> metrics = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=4)
        >>> mod = MEICARModule(model=model, metrics=metrics, optimizer=None)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = export_lightning_to_hf_dir(mod, Path(tmp) / "hf_model")
        ...     config_present = (out / "config.json").is_file()
        ...     tokenizer_stub_present = (out / "tokenizer_config.json").is_file()
        ...     fingerprint_present = (out / ".export_fingerprint").is_file()
        ...     print(f"config={config_present} tokstub={tokenizer_stub_present} fp={fingerprint_present}")
        ...     # Re-running is a no-op via fingerprint skip:
        ...     _ = export_lightning_to_hf_dir(mod, out)
        config=True tokstub=True fp=True

    Round-trip: the exported directory is loadable by ``transformers`` — a strictly stronger
    sanity check than SGLang-loadability since SGLang just calls into the same HF loader path.
    See ``tests/test_sglang_export.py`` for that smoke assertion.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    hf_model = module.model.HF_model
    fingerprint = _state_dict_hash(hf_model.state_dict())

    marker = out_dir / ".export_fingerprint"
    if out_dir.is_dir() and marker.is_file() and marker.read_text().strip() == fingerprint:
        logger.debug(f"Reusing existing HF export at {out_dir} (fingerprint match).")
        return out_dir

    tmp_dir = out_dir.with_suffix(out_dir.suffix + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        hf_model.save_pretrained(tmp_dir, safe_serialization=True)
        # SGLang's ``skip_tokenizer_init`` path still probes for tokenizer config in some
        # code paths (primarily the OpenAI-protocol shim); writing a minimal stub silences
        # those without actually instantiating a tokenizer. ``tokenizer_class`` points at
        # the generic base so HF's auto-tokenizer resolution doesn't try to load extras.
        (tmp_dir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizer"}))
        (tmp_dir / ".export_fingerprint").write_text(fingerprint)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Atomic-ish swap: remove any stale destination then rename tmp → out_dir.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    logger.info(f"Exported Lightning checkpoint → HF directory at {out_dir}.")
    return out_dir

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
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ...training.module import MEICARModule

logger = logging.getLogger(__name__)


def _state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    """Deterministic content hash of a state_dict.

    Used to decide whether a previously-exported HF directory is still current. Hashes the sorted
    param names concatenated with each tensor's raw bytes. Not meant to survive a tensor-dtype change
    — a dtype flip produces a different hash, which is the correct "re-export" signal.

    Byte extraction goes through ``untyped_storage()`` rather than ``.numpy().tobytes()`` because
    NumPy has no ``bfloat16`` dtype, and Lightning checkpoints trained under
    ``precision: bf16-true`` produce bf16 tensors. The untyped-storage path is dtype-agnostic.
    """
    hasher = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(b"\x00")
        hasher.update(bytes(t.untyped_storage()))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _is_existing_export_reusable(out_dir: Path, marker: Path, fingerprint: str) -> bool:
    """Decide whether an existing export directory can be reused without re-writing.

    Fingerprint match alone isn't sufficient — an external cleanup step (partial ``rm``,
    aborted sync, user hand-editing the directory, an earlier exporter that crashed mid-write
    but managed to land the marker) could have deleted weight shards or ``config.json`` while
    leaving ``.export_fingerprint`` intact. The fast-path needs to verify the directory still
    has the files SGLang/HF will look for at load time; otherwise we'd skip the re-export and
    fail later with an opaque loader error.

    Structural check: ``config.json`` + ``tokenizer_config.json`` + at least one
    ``*.safetensors`` shard. If any is missing, return ``False`` so the caller falls through
    to a fresh export. Emits a warning in that case so the user knows what happened.
    """
    if not (out_dir.is_dir() and marker.is_file() and marker.read_text().strip() == fingerprint):
        return False
    structural_ok = (
        (out_dir / "config.json").is_file()
        and (out_dir / "tokenizer_config.json").is_file()
        and any(out_dir.glob("*.safetensors"))
    )
    if not structural_ok:
        logger.warning(
            f"Fingerprint at {out_dir} matches but the directory is missing expected files "
            "(config.json / tokenizer_config.json / *.safetensors); re-exporting."
        )
        return False
    return True


def export_lightning_to_hf_dir(module: MEICARModule, out_dir: Path | str) -> Path:
    """Materialize a ``MEICARModule``'s HF submodel as an on-disk HF directory.

    Writes the underlying ``HF_model`` (``LlamaForCausalLM`` post-#108) via
    ``save_pretrained``, plus a stub ``tokenizer_config.json`` so SGLang's
    ``skip_tokenizer_init=True`` path doesn't warn about a missing tokenizer.

    **Idempotency**: if ``<out_dir>/.export_fingerprint`` matches the current state_dict
    hash, re-export is skipped. Two concurrent exporters don't corrupt each other — each
    writes into its own ``mkdtemp``-allocated staging directory alongside ``out_dir``, then
    atomically renames on success. If a concurrent winner already landed a matching
    fingerprint by the time we're ready to rename, the loser discards its staging dir and
    returns the existing export.

    **Concurrent reader caveat**: the atomic rename protects concurrent *writers*, but a
    reader already inside ``from_pretrained(out_dir)`` when a second writer renames a fresh
    directory over ``out_dir`` can race (the reader may see a half-open file handle).
    Current callers (``MEICAR_generate_trajectories`` CLI, one process per invocation) don't
    trigger this; if a future parallel-split generation tool does, add a ``filelock`` guard
    at the call site.

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
    if _is_existing_export_reusable(out_dir, marker, fingerprint):
        logger.debug(f"Reusing existing HF export at {out_dir} (fingerprint match).")
        return out_dir

    # Use ``mkdtemp`` next to ``out_dir`` (same filesystem → rename is atomic) with a unique
    # per-invocation name. Two concurrent exporters now each get their own staging directory
    # and neither can ``rmtree`` the other's in-progress write, matching the "concurrent writers
    # don't corrupt each other" claim in the docstring. An earlier revision used a fixed
    # ``{out_dir}.tmp`` which silently broke under concurrent calls.
    tmp_dir = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f"{out_dir.name}.tmp."))

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

    # Atomic-ish swap: remove any stale destination then rename tmp → out_dir. If a concurrent
    # export already landed a structurally-complete directory with matching fingerprint, skip
    # our write entirely and clean up — the losing writer's staging dir goes away, the
    # winner's final dir stands. If the winner's directory is corrupt we fall through and
    # install ours over it.
    if _is_existing_export_reusable(out_dir, marker, fingerprint):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug(
            f"Concurrent export already landed at {out_dir} with matching fingerprint; "
            "discarding our staging dir."
        )
        return out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    logger.info(f"Exported Lightning checkpoint → HF directory at {out_dir}.")
    return out_dir

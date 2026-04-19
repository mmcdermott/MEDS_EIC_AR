"""Smoke tests for the Lightning→HF-directory export (``backends.export``).

The SGLang backend loads weights via ``sglang.Engine(model_path=hf_dir)`` which under the
hood calls into the HF loader. If ``export_lightning_to_hf_dir`` produces a directory that
fails ``LlamaForCausalLM.from_pretrained``, SGLang will fail the same way — so a pure HF
round-trip is a strictly sufficient smoke check, and cheap enough to run in every CI lane
(no SGLang dep, no GPU).

These tests cover:

1. The export produces a directory with the expected files (config, tokenizer stub, safetensor
   weights, fingerprint marker).
2. The directory is reloadable via ``LlamaForCausalLM.from_pretrained`` and the reloaded
   state_dict matches the original param-for-param.
3. Re-export is idempotent — calling twice with unchanged weights skips the heavy write
   on the second call (detected by fingerprint-only read on the marker file).
4. Re-export after weight mutation re-writes, because the fingerprint changes.
5. Re-export also fires when *config-only* metadata changes (e.g. ``eos_token_id``
   auto-populated at runtime), even though the weights didn't change — the fingerprint
   folds in ``config.to_dict()`` alongside the state_dict.
6. Re-export fires when a fingerprint-matching output directory is structurally corrupt
   (missing ``*.safetensors`` shards, missing ``config.json``, etc.), so a partial-cleanup
   scenario doesn't leave an unloadable directory in place.

Why not test idempotency via timing: the first export is already fast enough at our scale
that wall-clock variance makes a timing assertion flaky. Instead we probe that the existing
fingerprint is *read* (via ``marker.read_text()``) rather than asserting mtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import LlamaConfig, LlamaForCausalLM

if TYPE_CHECKING:
    from pathlib import Path

from MEDS_EIC_AR.model import Model
from MEDS_EIC_AR.model.backends.export import export_lightning_to_hf_dir
from MEDS_EIC_AR.training.metrics import NextCodeMetrics
from MEDS_EIC_AR.training.module import MEICARModule


def _tiny_module() -> MEICARModule:
    _ = torch.manual_seed(0)
    model = Model(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 4,
            "max_position_embeddings": 10,
            "vocab_size": 10,
        }
    )
    metrics = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=4)
    return MEICARModule(model=model, metrics=metrics, optimizer=None)


def test_export_produces_loadable_hf_directory(tmp_path: Path):
    """End-to-end round-trip: export a tiny Lightning module, reload as a ``LlamaForCausalLM``,
    assert parameter tensors match the source.

    This is the primary load-bearing test. If it fails, any SGLang integration built on top
    fails the same way. It also catches the concrete gotcha called out in issue #88: a
    Llama-config edge case (e.g. mis-populated ``num_key_value_heads``, missing ``head_dim``)
    makes the directory unloadable.
    """
    mod = _tiny_module()
    out = export_lightning_to_hf_dir(mod, tmp_path / "hf_model")

    # File-layout sanity.
    assert (out / "config.json").is_file()
    assert (out / "tokenizer_config.json").is_file()
    assert (out / ".export_fingerprint").is_file()
    # save_pretrained with safe_serialization=True produces at least one .safetensors shard.
    assert list(out.glob("*.safetensors")), f"No safetensors shards found under {out}"

    # Reload via HF and compare tensors.
    reloaded_config = LlamaConfig.from_pretrained(out)
    reloaded = LlamaForCausalLM.from_pretrained(out)
    src = mod.model.HF_model

    assert reloaded_config.hidden_size == src.config.hidden_size
    assert reloaded_config.num_attention_heads == src.config.num_attention_heads
    assert reloaded_config.num_key_value_heads == src.config.num_key_value_heads
    assert reloaded_config.head_dim == src.config.head_dim
    assert reloaded_config.vocab_size == src.config.vocab_size

    src_sd = src.state_dict()
    reloaded_sd = reloaded.state_dict()
    assert set(src_sd) == set(reloaded_sd), (
        "Reloaded state_dict has different parameter names than the source. "
        f"Source extras: {set(src_sd) - set(reloaded_sd)}. "
        f"Reloaded extras: {set(reloaded_sd) - set(src_sd)}."
    )
    for name, t_src in src_sd.items():
        t_re = reloaded_sd[name]
        assert torch.equal(t_src, t_re), f"Parameter {name} differs after export/reload."


def test_export_is_idempotent_on_unchanged_weights(tmp_path: Path):
    """Re-exporting unchanged weights must be a no-op at the file level.

    Without idempotency, every ``MEICAR_generate_trajectories`` invocation would rewrite
    the whole HF directory even though the checkpoint hasn't moved — wasted I/O and a
    realistic source of "why is my disk filling up" complaints. The fingerprint marker is
    what enables the skip.
    """
    mod = _tiny_module()
    out = tmp_path / "hf_model"

    first = export_lightning_to_hf_dir(mod, out)
    # Capture fingerprint after first write.
    fp1 = (first / ".export_fingerprint").read_text()
    # Mutate a non-weight file to prove the skip path doesn't rewrite it.
    canary = first / "config.json"
    original_config = canary.read_text()
    canary.write_text(original_config + "\n# canary sentinel")

    second = export_lightning_to_hf_dir(mod, out)

    assert second == first
    fp2 = (second / ".export_fingerprint").read_text()
    assert fp1 == fp2  # same fingerprint → skip path ran
    # Canary survived — skip path did not touch config.json.
    assert "# canary sentinel" in canary.read_text()


def test_export_rewrites_after_weight_mutation(tmp_path: Path):
    """Mutating a source weight must change the fingerprint and trigger a re-write.

    Without this, silently-updated Lightning checkpoints would ship stale HF directories (e.g., during
    debugging when the user checkpoint-patches a weight and reruns). The fingerprint is content-based, so any
    param-tensor diff flips it.
    """
    mod = _tiny_module()
    out = tmp_path / "hf_model"

    export_lightning_to_hf_dir(mod, out)
    fp1 = (out / ".export_fingerprint").read_text()

    # Perturb one weight. Using ``with torch.no_grad()`` to avoid grad bookkeeping noise.
    with torch.no_grad():
        first_param = next(mod.model.HF_model.parameters())
        first_param.add_(1.0)

    export_lightning_to_hf_dir(mod, out)
    fp2 = (out / ".export_fingerprint").read_text()

    assert fp1 != fp2, (
        "Fingerprint did not change after weight mutation — re-export would silently reuse stale weights."
    )


def test_export_rewrites_when_config_changes_but_weights_do_not(tmp_path: Path):
    """Config mutations without weight changes must still trigger a re-export.

    Real scenario: ``MEICAR_generate_trajectories`` auto-populates
    ``hf_model.config.eos_token_id`` from the dataset if it's unset, mutating the config
    without touching any param tensors. A weights-only fingerprint would happily reuse a
    stale ``config.json`` with the old eos_token_id and break cross-chunk stopping in the
    rolling loop. The fingerprint now hashes ``config.to_json_string()`` alongside the
    state_dict to close that gap.
    """
    mod = _tiny_module()
    out = tmp_path / "hf_model"

    export_lightning_to_hf_dir(mod, out)
    fp1 = (out / ".export_fingerprint").read_text()

    # Mutate a config attribute without changing any weights.
    mod.model.HF_model.config.eos_token_id = 99

    export_lightning_to_hf_dir(mod, out)
    fp2 = (out / ".export_fingerprint").read_text()

    assert fp1 != fp2, (
        "Fingerprint did not change after config-only mutation — re-export would silently "
        "reuse a stale config.json (e.g., with the old eos_token_id)."
    )
    # Re-loaded config should carry the new value:
    from transformers import LlamaConfig

    reloaded = LlamaConfig.from_pretrained(out)
    assert reloaded.eos_token_id == 99


def test_export_rewrites_when_structurally_corrupt(tmp_path: Path):
    """A fingerprint-matching directory that's missing load-time files must trigger re-export.

    Reliability guard: an external cleanup step (partial ``rm``, aborted sync, user editing)
    could delete weight shards while leaving ``.export_fingerprint`` intact. The fast-path
    skip used to fire anyway, then SGLang/HF would fail at load time with an opaque error.
    Now the fast-path verifies structural integrity (config.json + tokenizer_config.json +
    at least one *.safetensors) before reusing; if any is missing, it re-exports.
    """
    mod = _tiny_module()
    out = tmp_path / "hf_model"

    first = export_lightning_to_hf_dir(mod, out)
    # Delete every safetensors shard but keep the fingerprint marker. A pure fingerprint check
    # would happily skip the re-export and leave the directory unloadable.
    shards = list(first.glob("*.safetensors"))
    assert shards, "test setup: initial export should produce at least one safetensors shard"
    for shard in shards:
        shard.unlink()
    assert (first / ".export_fingerprint").is_file()

    # Re-run: should detect the missing shard, warn, and re-export.
    export_lightning_to_hf_dir(mod, out)

    assert list(first.glob("*.safetensors")), (
        "Structural re-check should have triggered a re-export and restored the safetensors shards."
    )

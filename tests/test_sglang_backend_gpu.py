"""GPU-gated regression tests for issue #176 against a real ``sglang.Engine``.

The mock-based tests in ``test_sglang_backend.py`` and the scripted-backend tests in
``test_generate_trajectories.py`` pin the *logic* of the fix: the rolling loop never sends a
finished row to the backend, and the adapter refuses an all-pad row. What they cannot show is
that the real engine survives the sequence of batches the rolling loop actually produces, or
that SGLang still needs protecting at all. These tests do that on a tiny random-weight Llama
(``head_dim=64``, so SGLang's default FlashInfer attention path runs, as in the reported job).

Gated on ``sglang`` being importable and a CUDA device being present, like
``tests/grammar/test_cli_sglang.py``; CI collects-and-skips. Run locally with::

    PATH=$(pwd)/.venv/bin:$PATH uv run pytest tests/test_sglang_backend_gpu.py

Engine memory is kept small (``mem_fraction_static=0.08``, 4096 KV tokens) so the tests can
share a GPU with other work.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

pytest.importorskip("sglang")

from MEDS_EIC_AR.model.backends import SGLangBackend
from MEDS_EIC_AR.model.backends.sglang import _strip_padding_to_lists
from MEDS_EIC_AR.model.model import Model

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SGLang requires a CUDA GPU; skipping on CPU-only runners.",
)

VOCAB = 64
# Pad id for tensors the tests build directly.
PAD = 0
# Pad and EOS ids for the rolling test, chosen *outside* the model's vocabulary so the engine can
# never emit them: the loop strips any generated token equal to the pad id from the next window
# exactly as it strips padding, and a random-weight model under fp16 greedy decoding is not
# reproducible from call to call (radix-cache hits change the numeric path and near-tied argmaxes
# flip), so no in-vocabulary id can be trusted to stay unemitted. Pad positions are stripped
# before reaching the engine and EOS is injected by the test seam, so neither id is ever looked up.
ROLLING_EOS = VOCAB
ROLLING_PAD = VOCAB + 1
GPT_KWARGS = {
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "hidden_size": 128,  # head_dim = 64: at FlashInfer's floor, so the default attention backend runs
    "max_position_embeddings": 64,
    "vocab_size": VOCAB,
}
ENGINE_KWARGS = {
    "mem_fraction_static": 0.08,
    "max_total_tokens": 4096,
    "max_running_requests": 16,
    "disable_cuda_graph": True,
    "port": 31178,
}
# Four distinct 3-token prompts.
PROMPTS = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]], dtype=torch.long)
FULL_MASK = torch.ones_like(PROMPTS, dtype=torch.bool)


@pytest.fixture(scope="module")
def tiny_hf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export a seeded random-weight ``Model`` the way ``export_lightning_to_hf_dir`` would."""
    torch.manual_seed(0)
    model = Model(GPT_KWARGS, precision="32-true")
    # SGLang stops a request on the exported model's own EOS (``generation_config.json``, which
    # ``save_pretrained`` writes from the id captured at construction) as well as on the
    # ``stop_token_ids`` the adapter sends. ``MEICAR_pretrain`` passes the timeline-end token in at
    # construction, so both exported files agree with the rolling loop in production. A bare
    # ``Model`` would otherwise ship ``LlamaConfig``'s default of 2, an ordinary vocabulary id
    # here, and rows would stop on it without the rolling loop ever learning they had finished.
    model.HF_model.config.eos_token_id = ROLLING_EOS
    model.HF_model.generation_config.eos_token_id = ROLLING_EOS
    out = tmp_path_factory.mktemp("tiny_hf_model")
    model.HF_model.save_pretrained(out, safe_serialization=True)
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizer"}))
    return out


@pytest.fixture(scope="module")
def backend(tiny_hf_dir: Path):
    b = SGLangBackend(tiny_hf_dir, engine_kwargs=ENGINE_KWARGS)
    try:
        yield b
    finally:
        b.shutdown()


def _greedy_cfg(max_new_tokens: int, eos_token_id: int | None, pad_token_id: int = PAD) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id, eos_token_id=eos_token_id
    )


def _engine_is_alive(backend: SGLangBackend) -> bool:
    out = backend.generate_chunk(
        PROMPTS[:1], attention_mask=FULL_MASK[:1], generation_config=_greedy_cfg(2, None)
    )
    return tuple(out.shape) == (1, 2)


class _ForceEarlyEOS:
    """Pass-through to the real backend that makes row 0 finish on its first generated token.

    Records the stripped prompts of every call, i.e. exactly what the real adapter submits to
    the engine. Rewriting row 0's first chunk to ``EOS`` followed by pad is the only seam; the
    engine sees every batch the rolling loop produces, unmodified.
    """

    def __init__(self, real: SGLangBackend, eos: int):
        self.real, self.eos = real, eos
        self.sent: list[list[list[int]]] = []

    @property
    def max_prompt_len(self) -> int:
        return self.real.max_prompt_len

    def generate_chunk(self, input_ids, *, attention_mask, generation_config, **kwargs):
        self.sent.append(_strip_padding_to_lists(input_ids, attention_mask))
        out = self.real.generate_chunk(
            input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs
        )
        if len(self.sent) == 1:
            out[0, 0] = self.eos
            out[0, 1:] = generation_config.pad_token_id
        return out


def test_finished_rows_never_reach_the_engine(backend: SGLangBackend):
    """Rolling generation on the real engine, with one row finishing long before the others.

    Before the fix for #176, row 0's all-pad window from chunk 2 onward reached the engine as an empty prompt,
    which crashes the scheduler whenever such prompts form a prefill batch of their own. Here every call to
    the real engine must carry only unfinished rows and no empty prompt, the finished row must come back as
    EOS-then-pad in its own row, and the engine must still be usable afterwards.
    """
    torch.manual_seed(0)
    model = Model(GPT_KWARGS, precision="32-true")

    pad, eos = ROLLING_PAD, ROLLING_EOS
    model.HF_model.config.eos_token_id = eos

    spy = _ForceEarlyEOS(backend, eos)
    model.set_backend(spy)
    batch = Mock(code=PROMPTS, PAD_INDEX=pad, mode="SM")
    out = model._rolling_generate(batch, max_new_tokens=150, rolling_context_size=4, do_sample=False)

    assert len(spy.sent) >= 2, "budget must span several chunks for the finished-row path to be exercised"
    assert [len(p) for p in spy.sent[0]] == [3, 3, 3, 3]
    for i, prompts in enumerate(spy.sent[1:], start=2):
        assert len(prompts) < PROMPTS.shape[0], f"chunk {i} still carried the finished row"
        assert all(len(p) > 0 for p in prompts), f"chunk {i} submitted an empty prompt to the engine"
    assert all(len(prompts) == PROMPTS.shape[0] - 1 for prompts in spy.sent[1:]), "only row 0 ever finishes"

    assert out.shape[0] == PROMPTS.shape[0]
    assert out[0, 0].item() == eos and bool((out[0, 1:] == pad).all())
    assert bool((out[1:, 0] != pad).all()), "unfinished rows keep generating in their own rows"
    assert _engine_is_alive(backend)


def test_all_pad_row_is_refused_and_the_engine_survives(backend: SGLangBackend):
    """The adapter guard fires in-process, before the request can reach the scheduler."""
    input_ids = torch.tensor([[1, 2, 3], [PAD, PAD, PAD]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"rows \[1\] of this batch have no real \(non-pad\) tokens"):
        backend.generate_chunk(
            input_ids, attention_mask=input_ids != PAD, generation_config=_greedy_cfg(2, None)
        )
    assert _engine_is_alive(backend)


_CHILD = r"""
import json, sys
import sglang as sgl
kwargs = json.loads(sys.argv[2])
engine = sgl.Engine(model_path=sys.argv[1], skip_tokenizer_init=True, log_level="error", **kwargs)
try:
    out = engine.generate(input_ids=[[]], sampling_params={"max_new_tokens": 2, "temperature": 0.0})
    print("SILENT_OUTPUT_FOR_EMPTY_PROMPT:", out, flush=True)
finally:
    engine.shutdown()
"""


def test_sglang_does_not_silently_generate_from_an_empty_prompt(tiny_hf_dir: Path):
    """Canary for the upstream behaviour the adapter guard exists for.

    Sent a batch consisting of one empty prompt, ``sglang==0.5.9`` crashes its scheduler
    subprocess (RoPE: ``cannot reshape tensor of 0 elements``) and SIGQUITs the parent, so this
    runs in a subprocess. The property asserted is only that the engine never *returns tokens*
    for that prompt: a crash and an explicit rejection both pass. If a future SGLang starts
    returning output here, the guard in ``SGLangBackend.generate_chunk`` is still correct (there
    is nothing to condition on) but this test should be revisited.
    """
    kwargs = {**ENGINE_KWARGS, "port": ENGINE_KWARGS["port"] + 1}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tiny_hf_dir), json.dumps(kwargs)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = proc.stdout + proc.stderr
    assert "SILENT_OUTPUT_FOR_EMPTY_PROMPT" not in combined, (
        "SGLang returned tokens for an empty prompt; revisit the adapter guard's rationale.\n"
        + combined[-2000:]
    )
    assert proc.returncode != 0 or "Error" in combined, combined[-2000:]

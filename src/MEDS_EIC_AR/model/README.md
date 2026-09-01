# Core Model

This repository trains a Llama-style decoder-only transformer (RMSNorm + SwiGLU + full-dim RoPE, no biases)
over the code-vocabulary (after value quantization) of a MEDS dataset.

See `Model` in [`model.py`](model.py) for the thin wrapper that adapts `LlamaForCausalLM` to
`MEDSTorchBatch` inputs. The architecture-agnostic rolling-generation loop lives on the same class as
`Model._rolling_generate`. The pluggable generation-backend abstraction — the `GenerationBackend`
protocol and the default `HFBackend` — lives in [`backends/`](backends/).

## Capabilities

- **Rolling sliding-window generation** (PR #86) — `Model.generate` handles arbitrary `max_new_tokens`
    beyond the model's per-chunk capacity by sliding a context window and re-prompting with the tail of
    the running sequence until EOS or the step cap.
- **Pluggable generation backends** (PR #107, step 1 of #88) — an engine-agnostic `GenerationBackend`
    protocol lets non-HF inference engines (SGLang, vLLM) drop in behind the same interface. `HFBackend`
    is the default; `SGLangBackend` is in-flight (#117).
- **Llama config-snap at construction** — after the caller's `gpt_kwargs` overrides land on the
    vanilla `LlamaConfig`, `num_key_value_heads` is snapped to `num_attention_heads` and `head_dim` to
    `hidden_size // num_attention_heads` unless the caller set them explicitly. Plain MHA is the default;
    set `num_key_value_heads` to opt into GQA. Non-divisible combinations are rejected at construction
    time rather than surfacing as a cryptic shape error on the first forward pass.

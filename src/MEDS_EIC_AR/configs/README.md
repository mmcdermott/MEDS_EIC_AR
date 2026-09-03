# Configuration Files for MEICAR Model

This submodule contains configuration files for running the MEICAR model through the Hydra platform. The
configuration structure is as follows:

```python
>>> print_directory("./src/MEDS_EIC_AR/configs", config=PrintConfig(file_extension=".yaml"))
├── _demo_generate_trajectories.yaml
├── _demo_pretrain.yaml
├── _generate_trajectories.yaml
├── _pretrain.yaml
├── backend
│   ├── hf.yaml
│   ├── sglang.yaml
│   └── sglang_demo.yaml
├── datamodule
│   ├── default.yaml
│   ├── generate_trajectories.yaml
│   └── pretrain.yaml
├── inference
│   ├── default.yaml
│   └── demo.yaml
├── lightning_module
│   ├── LR_scheduler
│   │   └── get_cosine_schedule_with_warmup.yaml
│   ├── default.yaml
│   ├── demo.yaml
│   ├── large.yaml
│   ├── medium.yaml
│   ├── metrics
│   │   └── default.yaml
│   ├── micro.yaml
│   ├── model
│   │   ├── default.yaml
│   │   ├── demo.yaml
│   │   ├── large.yaml
│   │   ├── medium.yaml
│   │   ├── micro.yaml
│   │   └── small.yaml
│   ├── optimizer
│   │   └── adamw.yaml
│   └── small.yaml
└── trainer
    ├── callbacks
    │   ├── default.yaml
    │   ├── early_stopping.yaml
    │   ├── generation.yaml
    │   ├── generation_speed_logger.yaml
    │   ├── learning_rate_monitor.yaml
    │   ├── model_checkpoint.yaml
    │   └── save_logger_run_ids.yaml
    ├── default.yaml
    ├── demo.yaml
    ├── demo_generate.yaml
    ├── generate.yaml
    └── logger
        ├── csv.yaml
        ├── mlflow.yaml
        └── wandb.yaml

```

## Top-level configuration:

Two root configuration files drive the main entry points:

- `_pretrain.yaml` – used by `MEICAR_pretrain` to train the model. This
    config wires together the datamodule, lightning module and trainer
    configurations and exposes parameters such as `max_seq_len`, the output
    directory and whether training should resume from an existing run.
- `_generate_trajectories.yaml` – used by `MEICAR_generate_trajectories` to
    perform zero‑shot inference. It loads a pretrained model from
    `model_initialization_dir`, resolves the sequence length trade‑off between
    context and generation and again assembles the sub‑configs required for the
    datamodule and trainer.

Both of these files can be overridden on the command line or by providing a
custom YAML file to Hydra.

## `datamodule` configuration:

Configuration for constructing dataloaders backed by
`meds-torch-data`. `default.yaml` specifies common options such as the batch
size and how many workers to use. `pretrain.yaml` sets the random sampling
strategy and references `max_seq_len` from the top level while
`generate_trajectories.yaml` adjusts the sampling strategy so that the input
sequence is taken up to the prediction time and pads on the left.

## `backend` configuration:

Selects the inference engine used inside `Model._generate_chunk`. Each file names
a builder via `_target_` that Hydra instantiates at run time; adding a new
backend is one builder function + one yaml, no CLI code change.

- `hf.yaml` (default) — wraps HuggingFace's `generate` and is byte-identical
    to the original path (no engine dependency beyond `transformers`). Returns
    `None` from its builder, which tells the CLI to leave the model's default
    `HFBackend` in place.
- `sglang.yaml` — activates the SGLang engine adapter
    (`MEDS_EIC_AR.model.backends.SGLangBackend`, issue #88) with production
    defaults: CUDA graph capture enabled (amortizes over long batched-inference
    runs), `max_running_requests=256`. Requires the optional `sglang` extra and
    a CUDA device.
- `sglang_demo.yaml` — SGLang with test/demo defaults
    (`disable_cuda_graph=true`, `max_running_requests=8`). Trades peak decode
    throughput for fast cold-start, which is the right choice for short-lived
    runs (integration tests, one-shot debug generation) where CUDA-graph capture
    wouldn't pay for itself. Selected by the GPU-gated grammar test
    (`tests/grammar/test_cli_sglang.py`).

Pick one with `backend=<name>` on the CLI. Production SGLang inference uses
`backend=sglang`; the gated integration test uses `backend=sglang_demo`.

## `inference` configuration:

Settings that control zero‑shot trajectory generation. The default configuration
specifies which dataset splits to run over (`tuning` and `held_out`) and how many
trajectories to sample per task example.

## `lightning_module` configuration:

Defines the Lightning module and all of its constituent parts. The `model`
subdirectory contains presets for different model sizes while `optimizer`,
`LR_scheduler` and `metrics` hold their respective configuration objects. The
top level file ties these pieces together so that the module can be instantiated
by Hydra.

## `trainer` configuration:

Options passed directly to `lightning.pytorch.Trainer`. `default.yaml` provides
reasonable defaults for mixed precision training, logging frequency and
gradient clipping. The `callbacks` and `logger` subdirectories contain reusable
definitions for common callbacks such as model checkpointing and CSV, MLFlow or
WandB logging.

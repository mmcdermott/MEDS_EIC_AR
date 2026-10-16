# Lightning Modules

[`MEICARModule`](module.py) wraps the core [`Model`](../model/model.py) in a Lightning module for training,
validation, and prediction. It owns:

- **Metrics** — [`NextCodeMetrics`](metrics.py): a `torchmetrics.MetricCollection` of top-$k$ accuracy and
    perplexity over next-code prediction. Logged per-step on training and per-epoch on validation with
    `sync_dist=True` for multi-device runs.
- **Optimizer + LR scheduler** — instantiated from Hydra `partial`s so the model parameters can be bound
    at training time. AdamW is the default; `torch.optim.lr_scheduler` or `transformers` schedulers are
    supported (see [`configs/lightning_module/`](../configs/lightning_module/)). A no-weight-decay param
    group is built for all norm / bias parameters via [`_norm_bias_param_names`](module.py).
- **Resume-safe config persistence** — `save_hyperparameters` plus a resolved-config write-back to the run
    dir so resumes can verify no load-bearing param changed. A small allow-list
    (`ALLOWED_DIFFERENCE_KEYS` in [`files.py`](files.py)) covers cadence params that are safe to adjust
    across a resume (`trainer.val_check_interval`, `trainer.check_val_every_n_epoch`,
    `trainer.log_every_n_steps`). The step budget (`trainer.max_steps`, `trainer.max_epochs`) is also
    allowed to change, but only when both runs use a horizon-free LR schedule
    (`HORIZON_FREE_LR_SCHEDULERS`).
- **Anytime EMA** — [`AnytimeEMA`](anytime.py) keeps a growing-horizon average of the weights
    (`beta_t = 2^(-k / t)`), updated once per optimizer step. Validation swaps the EMA weights in and then
    restores the raw weights exactly. The EMA and its age are checkpointed as callback state and checked
    against the restored step on resume. `MEICARModule.load_from_checkpoint(..., weights="auto")` loads the
    EMA weights whenever a checkpoint has them.
- **Prediction** — `predict_step` takes a `PredictBatch` — the
    `(mdata_batch, dataset_row_idxs, trajectory_idxs)` three-tuple yielded by the N-sample
    interleaving dataloader in [`generation/`](../generation/) — and runs the rolling generator
    to produce N trajectories per base-dataset row in a single predict pass. For a task-labeled
    `MEDSPytorchDataset` that is one row per `(subject_id, prediction_time)` pair — a subject
    with multiple prediction times contributes N trajectories at each of them.

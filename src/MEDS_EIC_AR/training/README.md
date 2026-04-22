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
    `trainer.log_every_n_steps`).
- **Prediction** — `predict_step` takes a `PredictBatch` (`(mdata_batch, subject_idxs, trajectory_idxs)`
    from the N-sample interleaving dataloader, see [`generation/`](../generation/)) and runs the rolling
    generator to produce N trajectories per subject in a single predict pass.

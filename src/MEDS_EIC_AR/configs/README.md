# Configuration Files for MEICAR Model

This submodule contains configuration files for running the MEICAR model through the Hydra platform. The
configuration structure is as follows:

```python
>>> print_directory("./src/MEDS_EIC_AR/configs", config=PrintConfig(file_extension=".yaml"))
├── _demo_generate_trajectories.yaml
├── _demo_pretrain.yaml
├── _generate_trajectories.yaml
├── _pretrain.yaml
├── datamodule
│   ├── default.yaml
│   ├── generate_trajectories.yaml
│   └── pretrain.yaml
├── inference
│   ├── default.yaml
│   └── demo.yaml
├── lightning_module
│   ├── LR_scheduler
│   │   ├── cosine_annealing_warm_restarts.yaml
│   │   ├── get_cosine_schedule_with_warmup.yaml
│   │   ├── one_cycle_LR.yaml
│   │   └── reduce_LR_on_plateau.yaml
│   ├── default.yaml
│   ├── demo.yaml
│   ├── metrics
│   │   └── default.yaml
│   ├── model
│   │   ├── default.yaml
│   │   ├── demo.yaml
│   │   └── small.yaml
│   └── optimizer
│       ├── adam.yaml
│       └── adamw.yaml
└── trainer
    ├── callbacks
    │   ├── default.yaml
    │   ├── early_stopping.yaml
    │   ├── learning_rate_monitor.yaml
    │   └── model_checkpoint.yaml
    ├── default.yaml
    ├── demo.yaml
    └── logger
        ├── csv.yaml
        ├── mlflow.yaml
        └── wandb.yaml

```

## Top-level configuration:

TODO

## `datamodule` configuration:

TODO

## `inference` configuration:

TODO

## `lightning_module` configuration:

TODO

## `trainer` configuration:

TODO

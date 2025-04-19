# MEDS "Everything-is-code" Autoregressive Model

[![PyPI - Version](https://img.shields.io/pypi/v/MEDS-EIC-AR)](https://pypi.org/project/MEDS-EIC-AR/)
![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)
[![codecov](https://codecov.io/gh/mmcdermott/MEDS_EIC_AR/graph/badge.svg?token=5RORKQOZF9)](https://codecov.io/gh/mmcdermott/MEDS_EIC_AR)
[![tests](https://github.com/mmcdermott/MEDS_EIC_AR/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_EIC_AR/actions/workflows/tests.yml)
[![code-quality](https://github.com/mmcdermott/MEDS_EIC_AR/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_EIC_AR/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/MEDS_EIC_AR#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/MEDS_EIC_AR/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/MEDS_EIC_AR.svg)](https://github.com/mmcdermott/MEDS_EIC_AR/graphs/contributors)

A MEDS, "Everything-is-code" style Autoregressive Generative Model, capable of zero-shot inference.

This is based on the [MEDS-Torch](https://github.com/Oufattole/meds-torch) model of the same name.

## Installation

```bash
pip install MEDS-EIC-AR
```

## Usage

### 1. Pre-process your data

You have three directories:

1. `$RAW_MEDS_DIR` -- The raw MEDS data directory that you want to pre-process.
2. `$INTERMEDIATE_DIR` -- An intermediate directory where the partially processed data will be stored prior
    to tokenization and tensorization.
3. `$FINAL_DATA_DIR` -- The final output directory where the tokenized and tensorized data will be stored.
    This directory is suitable for use in loading the data with `meds-torch-data`.

Run:

```bash
MEICAR_process_data input_dir="$RAW_MEDS_DIR" \
    intermediate_dir="$INTERMEDIATE_DIR" \
    output_dir="$FINAL_DATA_DIR"
```

You can also run this in demo mode, which lowers the filtering thresholds significantly so the script does not
filter out all data:

```bash
MEICAR_process_data ... do_demo=True
```

You can exert more fine-grained control on the filtering with the following environment variables:

1. `MIN_SUBJECTS_PER_CODE`: How many subjects must a given code be observed within to be included in the
    final vocabulary? Note that this excludes some sentinel codes which are always retained.
2. `MIN_EVENTS_PER_SUBJECT`: How many events must a subject have to be included in the final dataset?

> [!WARNING]
> I suspect this is not actually working yet. Tests currently just ensure it does not crash; not that the
> entire output of the pipeline looks as expected.

### 2. Pre-train the model

You can pre-train the model using the `MEICAR_pretrain` command. To use this, let us assume you have a new
directory to store the pretrained model artifacts called `$PRETRAINED_MODEL_DIR`. Then, you can run:

```bash
MEICAR_pretrain datamodule.config.tensorized_cohort_dir="$FINAL_DATA_DIR" \
    output_dir="$PRETRAINED_MODEL_DIR" \
    datamodule.batch_size=32 \
    trainer.max_epochs=10
```

to train the model for 10 epochs.

This uses a [Hydra](https://hydra.cc/) configuration system, with the root config located in the
[`_pretrain.yaml`](src/MEDS_EIC_AR/configs/_pretrain.yaml) file. You can override any of the nested
configuration parameters (as shown above via `datamodule.config.tensorized_cohort_dir` on the command line,
though you will more likely materialize an experimental configuration file to disk in yaml form and overwrite
the config path and name directly in the normal hydra manner.

> [!WARNING]
> I suspect this is not actually working yet. Tests currently just ensure it does not crash; not that the
> entire output of the pipeline looks as expected.

### 3. Zero-shot Inference

Zero-shot inference consists of two steps:

1. Given a task cohort and a pre-trained model, for each sample in the task cohort, generate future
    trajectories from those inputs forward with the pre-trained model and save them to disk in a pseudo-MEDS
    format.
2. Resolve these generated trajectories into concrete, probabilistic predictions for the task cohort.

#### 3.1 Generate Trajectories for a task spec.

You can directly generate trajectories using the `MEICAR_generate_trajectories` command. This requires a few
more configuration parameters than the pre-training step, so let's go through those:

1. You need to specify the task labels directory in the `datamodule.config.task_labels_dir` parameter.
2. You need to specify the model initialization directory in the `model_initialization_dir` parameter. This
    is the output directory of the pre-train step.
3. You need to specify how you want to trade-off between allowed input context size and the maximum possible
    generated trajectory length. The former allows you to use more of the patient's record, but the latter
    controls how far into the future you can predict. This can be configured with one of three parameters in
    the `seq_lens` part of the config. If you set:
    - `seq_lens.generation_context_size`, that will be the maximum length of the input context, and the
        remaining length of the pretrained model's maximum sequence length will be used for generation.
    - `seq_lens.max_generated_trajectory_len`, that will be the maximum length of the generated trajectory,
        and the remaining length of the pretrained model's maximum sequence length will be used for the
        input.
    - `seq_lens.frac_seq_len_as_context`, that will be the fraction of the pretrained model's maximum
        sequence length that will be used for the input context, and the remaining length will be used for
        generation. This is set by default to 0.25, which means that 25% of the maximum sequence length will
        be used for the input context, and 75% will be used for generation. If you wish to use another mode
        on the command line, be sure to set this to `null` to disable it.

With that in mind, you can run the following command to generate trajectories for a task cohort:

```bash
MEICAR_generate_trajectories \
    output_dir="$GENERATED_TRAJECTORIES_DIR" \
    model_initialization_dir="$PRETRAINED_MODEL_DIR" \
    datamodule.config.tensorized_cohort_dir="$FINAL_DATA_DIR" \
    datamodule.config.task_labels_dir="$TASK_ROOT_DIR/$TASK_NAME" \
    datamodule.batch_size=32
```

Not yet implemented.

#### 3.2 Resolve Trajectories into Predictions.

Not yet implemented.

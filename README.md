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
    model_dir="$PRETRAINED_MODEL_DIR" \
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

Not yet implemented.

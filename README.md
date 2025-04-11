# MEDS "Everything-is-code" Autoregressive Model

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
3. `$OUTPUT_DIR` -- The final output directory where the tokenized and tensorized data will be stored. This
    directory is suitable for use in loading the data with `meds-torch-data`.

Run:

```bash
MEICAR_process_data input_dir="$RAW_MEDS_DIR" intermediate_dir="$INTERMEDIATE_DIR" output_dir="$OUTPUT_DIR"
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

### 2. Train the model

Not yet implemented.

### 3. Zero-shot Inference

Not yet implemented.

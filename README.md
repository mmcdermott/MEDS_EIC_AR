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

### 2. Train the model

Not yet implemented.

### 3. Zero-shot Inference

Not yet implemented.

## Differences from the MEDS-Torch version

1. In comparison to the MEDS-Torch
    [config](https://github.com/Oufattole/meds-torch/blob/d1650ea6152301a9b9bdbd32756337214e5f310f/ZERO_SHOT_TUTORIAL/configs/eic_config.yaml),
    this version removes the "reorder measurements" pre-processing step and the additional code filtering
    that is specific to MIMIC.
2. The
    [`custom_time_token.py`](https://github.com/Oufattole/meds-torch/blob/d1650ea6152301a9b9bdbd32756337214e5f310f/src/meds_torch/utils/custom_time_token.py)
    script has been renamed to
    [`add_time_interval_tokens.py`](src/MEDS_EIC_AR/stages/add_time_interval_tokens.py) and exported as a
    [script](pyproject.toml) in this package.
3. The `custom_filter_measurements` script is replaced with the generic filter measurements augmented with
    the match and revise syntax.
4. The
    [`quantile_binning.py`](https://github.com/Oufattole/meds-torch/blob/d1650ea6152301a9b9bdbd32756337214e5f310f/src/meds_torch/utils/quantile_binning.py)
    script has been copied to
    [`quantile_binning.py`](src/MEDS_EIC_AR/stages/quantile_binning.py) and exported as a
    [script](pyproject.toml) in this package.
5. The list of included stages has been revised to include only the ones that are uniquely relevant to this
    model. Data tokenization and tensorization stages have been outsourced to
    [meds-torch-data](https://meds-torch-data.readthedocs.io/en/latest/)

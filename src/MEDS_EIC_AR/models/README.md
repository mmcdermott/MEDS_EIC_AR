# Core Model

This repository trains a GPT-NeoX like model over the code-vocabulary (after value quantization) of a MEDS
dataset.

## Differences from `meds-torch` core:

1. The model has been dramatically simplified and extracted from other aspects of the code. Lightning module
    integration and generation have been stripped out, as has the extra projection head atop code logit
    generation.
2. The separate input encoder has been removed, given the GPT-NeoX architecture takes codes as inputs
    directly and handles the embedding manually (and temporal position encodings were not used in the
    `meds-torch` base model).
3. The backbone and "model" portions of the code have been merged, as functionally the entire stack here is
    just a wrapper to ensure that the HF model can accept `MEDSTorchBatch` objects.

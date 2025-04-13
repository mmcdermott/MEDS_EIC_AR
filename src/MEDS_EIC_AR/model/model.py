import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

try:
    import flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class Model(torch.nn.Module):
    """A basic GPT-NeoX like model for pre-training an autoregressive, "everything-is-code" model.

    This model is a wrapper around the Hugging Face GPTNeoXForCausalLM model to run it over MEDS-TorchData
    batches.

    Args:
        gpt_kwargs: A dictionary of keyword arguments to pass to the GPTNeoXConfig constructor. These can
            include 'max_position_embeddings', 'vocab_size', 'hidden_size', etc.

    Examples:
        >>> import polars as pl
        >>> metadata_df = pl.read_parquet(preprocessed_dataset / "metadata" / "codes.parquet")
        >>> vocab_size = metadata_df.select(pl.col("code/vocab_index")).max().item() + 1
        >>> print(f"Vocab size: {vocab_size}")
        Vocab size: 38
        >>> _ = torch.manual_seed(0)
        >>> model = Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 2,
        ...     "hidden_size": 4,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": vocab_size,
        ... })
        >>> loss, outputs = model(sample_batch)
        >>> print(loss)
        tensor(3.6152, dtype=torch.float16, grad_fn=<NllLoss2DBackward0>)
        >>> print(f"Outputs have keys: {', '.join(outputs.keys())}")
        Outputs have keys: logits, past_key_values
        >>> print(f"Logits shape: {outputs.logits.shape}")
        Logits shape: torch.Size([2, 8, 38])
        >>> print(outputs.logits)
        tensor([[[ 3.4485e-02, ..., -2.7084e-02], ..., [ 7.9407e-02, ..., -2.3972e-02]],
        <BLANKLINE>
                [[ 3.3356e-02, ..., -2.7267e-02], ..., [ 7.9529e-02, ..., -2.4506e-02]]],
               dtype=torch.float16,
               grad_fn=<UnsafeViewBackward0>)
    """

    HF_model_config: GPTNeoXConfig
    HF_model: GPTNeoXForCausalLM

    def __init__(self, gpt_kwargs: dict | DictConfig):
        super().__init__()

        self.HF_model_config: GPTNeoXConfig = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")

        for key, val in gpt_kwargs.items():
            if not hasattr(self.HF_model_config, key):
                raise ValueError(f"Config for HF model gpt-neox does not have attribute {key}")
            setattr(self.HF_model_config, key, val)

        self.HF_model_config.intermediate_size = 4 * self.HF_model_config.hidden_size

        if HAS_FLASH_ATTN:
            self.HF_model = AutoModelForCausalLM.from_config(
                self.HF_model_config, attn_implementation="flash_attention_2"
            )
        else:
            self.HF_model = AutoModelForCausalLM.from_config(self.HF_model_config)

    def forward(self, batch: MEDSTorchBatch):
        outputs = self.HF_model(input_ids=batch.code, attention_mask=(batch.code == batch.PAD_INDEX))
        loss = F.cross_entropy(
            outputs.logits[:, :-1].transpose(2, 1), batch.code[:, 1:], ignore_index=batch.PAD_INDEX
        )

        return loss, outputs

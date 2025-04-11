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


class Model(torch.nn.Module):
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

        self.HF_model = AutoModelForCausalLM.from_config(
            self.HF_model_config, attn_implementation="flash_attention_2"
        )

    def forward(self, batch: MEDSTorchBatch):
        outputs = self.HF_model(input_ids=batch.code, attention_mask=(batch.code == batch.PAD_INDEX))
        loss = F.cross_entropy(outputs.logits[:, :-1], batch.code[:, 1:], ignore_index=batch.PAD_INDEX)

        return loss, outputs

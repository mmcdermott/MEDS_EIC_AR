import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

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
        >>> _ = torch.manual_seed(0)
        >>> model = Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 2,
        ...     "hidden_size": 4,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": dataset_config.vocab_size,
        ... })
        >>> model.max_seq_len
        10
        >>> model.vocab_size
        39
        >>> loss, outputs = model(sample_batch)
        >>> print(loss)
        tensor(3.6660, dtype=torch.float16, grad_fn=<NllLoss2DBackward0>)
        >>> print(f"Outputs have keys: {', '.join(outputs.keys())}")
        Outputs have keys: logits, past_key_values
        >>> print(f"Logits shape: {outputs.logits.shape}")
        Logits shape: torch.Size([2, 9, 39])
        >>> print(outputs.logits)
        tensor([[[ 0.0197, ...,  0.0197], ..., [ 0.0138, ...,  0.0166]],
        <BLANKLINE>
                [[ 0.0203,  ...,  0.0193], ..., [ 0.0145, ...,  0.0163]]],
               dtype=torch.float16,
               grad_fn=<UnsafeViewBackward0>)
        >>> sample_param_name, sample_param = next(iter(model.named_parameters()))
        >>> print(f"{sample_param_name} ({sample_param.shape}): {sample_param}")
        HF_model.gpt_neox.embed_in.weight (torch.Size([39, 4])): Parameter containing:
        tensor([[-0.0247, -0.0222,  0.0160,  0.0219],
                ...,
                [-0.0050, -0.0061, -0.0358,  0.0136]],
               dtype=torch.float16,
               requires_grad=True)
        >>> print(f"Sample parameter grad?: {sample_param.grad}")
        Sample parameter grad?: None
        >>> loss.backward()
        >>> print(f"Sample parameter grad?: {sample_param.grad}")
        Sample parameter grad?:
        tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                ...,
                [ 6.4240e-03, -5.7495e-02, -2.8915e-02,  7.9956e-02]],
               dtype=torch.float16)
    """

    HF_model_config: GPTNeoXConfig
    HF_model: GPTNeoXForCausalLM
    do_demo: bool

    def __init__(self, gpt_kwargs: dict | DictConfig, do_demo: bool = False):
        super().__init__()

        self.HF_model_config: GPTNeoXConfig = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")

        for key, val in gpt_kwargs.items():
            if not hasattr(self.HF_model_config, key):
                raise ValueError(f"Config for HF model gpt-neox does not have attribute {key}")
            setattr(self.HF_model_config, key, val)

        if HAS_FLASH_ATTN:
            self.HF_model = AutoModelForCausalLM.from_config(
                self.HF_model_config, attn_implementation="flash_attention_2"
            )
        else:
            self.HF_model = AutoModelForCausalLM.from_config(self.HF_model_config)

        self.do_demo = do_demo
        if self.do_demo:
            self.forward = self._forward_demo
        else:
            self.forward = self._forward

    @property
    def max_seq_len(self) -> int:
        """The maximum sequence length of the model."""
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        """The vocabulary size of the model."""
        return self.HF_model_config.vocab_size

    def _check_inputs(self, batch: MEDSTorchBatch):
        """Checks the inputs for various validity properties.

        Validity checks:
          - The batch is in "SM" mode (see
            [MEDSTorchBatch](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/types/#meds_torchdata.types.MEDSTorchBatch)
            for more details).
          - The input sequence length must not exceed the model's maximum sequence length.
          - The input sequence length must not be too short (minimum sequence length is 2).
          - The input must not contain out-of-vocabulary tokens.
          - The input must not contain inf or nan values.
          - The input must not contain only padding tokens.

        Args:
            batch: The input batch of data.

        Raises:
            ValueError: If the input sequence length exceeds the model's maximum sequence length.
            AssertionError: If the input contains out-of-vocabulary tokens or if it contains inf or nan
                values.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> batch = Mock(code=torch.LongTensor([[0, 3, 1], [0, 2, 1]]), PAD_INDEX=0, mode="SM")
            >>> model._check_inputs(batch) # no errors
            >>> batch.code = torch.LongTensor([[0, 3, 1], [0, 2, 11]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: Input sequence contains 1 out-of-vocabulary tokens (max 11 for vocab size 10).
            >>> batch.code = torch.Tensor([[0, 3, 1], [0, 2, float("inf")]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: Batch code contains inf values.
            >>> batch.code = torch.Tensor([[0, 3, 1], [0, 2, float("nan")]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: Batch code contains nan values.
            >>> batch.code = torch.LongTensor([[0, 3, 1], [0, 0, 0]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: 1 samples in the batch have only padding tokens. Batch size: 2, Sequence length: 3
            >>> batch.code = torch.LongTensor([[0, 3, 1, 0], [0, 2, 1, 4]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            ValueError: Input sequence length 4 exceeds model max sequence length 3.
            >>> batch.code = torch.LongTensor([[1], [2]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            ValueError: Input sequence length 1 is too short. Minimum sequence length is 2.
            >>> model._check_inputs(Mock(mode="SEM"))
            Traceback (most recent call last):
                ...
            ValueError: Batch mode SEM is not supported.
        """

        code = batch.code

        if batch.mode != "SM":
            raise ValueError(f"Batch mode {batch.mode} is not supported.")

        batch_size, seq_len = code.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {batch.code.shape[1]} exceeds model max sequence length "
                f"{self.max_seq_len}."
            )
        elif seq_len <= 1:
            raise ValueError(
                f"Input sequence length {batch.code.shape[1]} is too short. Minimum sequence length is 2."
            )

        torch._assert(~torch.isinf(code).any(), "Batch code contains inf values.")
        torch._assert(~torch.isnan(code).any(), "Batch code contains nan values.")

        out_of_vocab = code >= self.vocab_size
        out_of_vocab_msg = (
            f"Input sequence contains {out_of_vocab.sum()} out-of-vocabulary tokens "
            f"(max {batch.code.max()} for vocab size {self.vocab_size})."
        )

        torch._assert(~out_of_vocab.any(), out_of_vocab_msg)

        is_pad = code == batch.PAD_INDEX  # Shape is batch_size x seq_len
        all_samples_pad = is_pad.all(dim=1)  # Shape is batch_size

        all_samples_pad_msg = (
            f"{all_samples_pad.sum()} samples in the batch have only padding tokens. "
            f"Batch size: {code.shape[0]}, Sequence length: {code.shape[1]}"
        )
        torch._assert(~all_samples_pad.any(), all_samples_pad_msg)

    def _check_outputs(self, loss: torch.FloatTensor):
        """Checks the outputs for various validity properties.

        Validity checks:
            - The loss is not inf or nan.

        Args:
            loss: The loss tensor.

        Raises:
            AssertionError: If the loss contains inf or nan values.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> model._check_outputs(torch.FloatTensor([[0.1, 0.2], [0.3, 0.4]])) # no errors
            >>> model._check_outputs(torch.FloatTensor([[0.1, float("inf")], [0.3, 0.4]]))
            Traceback (most recent call last):
                ...
            AssertionError: Loss contains inf values.
            >>> model._check_outputs(torch.FloatTensor([[0.1, float("nan")], [0.3, 0.4]]))
            Traceback (most recent call last):
                ...
            AssertionError: Loss contains nan values.
        """
        torch._assert(~torch.isinf(loss).any(), "Loss contains inf values.")
        torch._assert(~torch.isnan(loss).any(), "Loss contains nan values.")

    def _forward_demo(self, batch: MEDSTorchBatch) -> tuple[torch.FloatTensor, CausalLMOutputWithPast]:
        """A demo forward pass that adds more checks and assertions."""

        self._check_inputs(batch)
        out = self._forward(batch)
        self._check_outputs(out[0])

        return out

    def _forward(self, batch: MEDSTorchBatch) -> tuple[torch.FloatTensor, CausalLMOutputWithPast]:
        outputs = self.HF_model(input_ids=batch.code, attention_mask=(batch.code == batch.PAD_INDEX))
        loss = F.cross_entropy(
            outputs.logits[:, :-1].transpose(2, 1), batch.code[:, 1:], ignore_index=batch.PAD_INDEX
        )

        return loss, outputs

    def generate(self, batch: MEDSTorchBatch, **kwargs) -> torch.Tensor:
        inputs = batch.code
        attention_mask = inputs == batch.PAD_INDEX

        generation_config = GenerationConfig(
            max_length=self.max_seq_len,
            do_sample=True,
            num_beams=1,  # no beam search
            temperature=1.0,
            pad_token_id=batch.PAD_INDEX,
        )

        return self.HF_model.generate(
            inputs,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )

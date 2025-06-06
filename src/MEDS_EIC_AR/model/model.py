import logging
from typing import ClassVar

import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
    logger.info("FlashAttention is available.")
except ImportError:
    HAS_FLASH_ATTN = False


def _val(tensor: torch.Tensor) -> int | bool | float:
    """Returns the value of a scalar-tensor as a Python scalar.

    Examples:
        >>> _val(torch.tensor(1))
        1
        >>> _val(torch.tensor(1.0))
        1.0
        >>> _val(torch.tensor(False))
        False
    """
    return tensor.detach().cpu().item()


class Model(torch.nn.Module):
    """A basic GPT-NeoX like model for pre-training an autoregressive, "everything-is-code" model.

    This model is a wrapper around the Hugging Face GPTNeoXForCausalLM model to run it over MEDS-TorchData
    batches.

    Args:
        gpt_kwargs: A dictionary of keyword arguments to pass to the GPTNeoXConfig constructor. These can
            include 'max_position_embeddings', 'vocab_size', 'hidden_size', etc.

    Raises:
        ValueError: If the gpt_kwargs contains a key that is not supported.

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> model = Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 2,
        ...     "hidden_size": 4,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": dataset_config.vocab_size,
        ... }, precision="16-true")

    Once created, we can use the simple helpers `max_seq_len` and `vocab_size` to access the corresponding
    elements of the model config:

        >>> model.max_seq_len
        10
        >>> model.vocab_size
        39

    We can also run the model on a sample batch of data. This `sample_batch` is defined in our `conftest.py`
    file and is a MEDSTorchBatch object. The only feature of this batch that we use in this model is the
    `code` key.

        >>> sample_batch
        MEDSTorchBatch(code=tensor([[38,  5, 36,  3, 13, 29, 35,  4, 37],
                                    [38,  5, 36,  2, 22, 25, 35,  4, 37]]),
                       ...)

    We run over the batch in the normal way, which internally calls the `forward` method of the model:

        >>> loss, outputs = model(sample_batch)
        >>> print(loss)
        tensor(3.6660, dtype=torch.float16, grad_fn=<NllLoss2DBackward0>)
        >>> print(f"Outputs have keys: {', '.join(outputs.keys())}")
        Outputs have keys: logits, past_key_values
        >>> print(f"Logits shape: {outputs.logits.shape}")
        Logits shape: torch.Size([2, 9, 39])
        >>> print(outputs.logits)
        tensor([[[ 2.0309e-02, ...,  1.9135e-02], ..., [ 1.3763e-02, ...,  1.6571e-02]],
        <BLANKLINE>
                [[ 2.0309e-02, ...,  1.9135e-02], ..., [ 1.4458e-02, ...,  1.6281e-02]]],
               dtype=torch.float16,
               grad_fn=<UnsafeViewBackward0>)

    The models parameters can be accessed in the normal way.

        >>> sample_param_name, sample_param = next(iter(model.named_parameters()))
        >>> print(f"{sample_param_name} ({sample_param.shape}): {sample_param}")
        HF_model.gpt_neox.embed_in.weight (torch.Size([39, 4])): Parameter containing:
        tensor([[-0.0247, -0.0222,  0.0160,  0.0219], ..., [-0.0050, -0.0061, -0.0358,  0.0136]],
               dtype=torch.float16,
               requires_grad=True)

    Let's validate that they have gradients that can be realized via `.backward()` as normal:

        >>> print(f"Sample parameter grad?: {sample_param.grad}")
        Sample parameter grad?: None
        >>> loss.backward()
        >>> print(f"Sample parameter grad?: {sample_param.grad}")
        Sample parameter grad?:
        tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                ...,
                [ 1.5251e-02, -6.2347e-02, -3.1921e-02,  7.9102e-02]],
               dtype=torch.float16)

    With a single backward pass, we should not get any infinite gradients:

        >>> for name, param in model.named_parameters():
        ...     if param.grad is not None:
        ...         if not _val(torch.isfinite(param.grad).all()):
        ...             raise ValueError(f"Gradient for {name} is not finite.")

    Model errors are raised if we pass invalid GPT kwargs to the constructor:

        >>> Model({"foobar": 2})
        Traceback (most recent call last):
            ...
        ValueError: Config for HF model gpt-neox does not have attribute foobar
    """

    HF_model_config: GPTNeoXConfig
    HF_model: GPTNeoXForCausalLM
    do_demo: bool
    precision: str

    PRECISION_TO_MODEL_WEIGHTS_DTYPE: ClassVar[dict[str, torch.dtype]] = {
        "32-true": torch.float32,
        "16-true": torch.float16,
        "16-mixed": torch.float32,
        "bf16-true": torch.bfloat16,
        "bf16-mixed": torch.float32,
        "transformer-engine": torch.bfloat16,
    }

    def __init__(self, gpt_kwargs: dict | DictConfig, precision: str = "32-true", do_demo: bool = False):
        super().__init__()

        self.HF_model_config: GPTNeoXConfig = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")

        for key, val in gpt_kwargs.items():
            if key == "attention_head_dim":
                continue
            if not hasattr(self.HF_model_config, key):
                raise ValueError(f"Config for HF model gpt-neox does not have attribute {key}")
            setattr(self.HF_model_config, key, val)

        extra_kwargs = {"torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}

        if HAS_FLASH_ATTN:
            logger.info("Using FlashAttention 2 for the model.")
            extra_kwargs["attn_implementation"] = "flash_attention_2"

            if precision in {"16-mixed", "bf16-mixed"}:
                logger.info(
                    "Using mixed precision for Flash Attention 2.0. Ignore the warning that may appear "
                    "below about Flash Attention 2.0 only supporting torch.float16 and torch.bfloat16. "
                    "Lightning will automatically cast the model to the correct dtype during training in "
                    "mixed precision mode."
                )
            elif precision not in {"16-true", "bf16-true", "transformer-engine"}:
                logger.warning(
                    "Flash Attention 2.0 is only supported for precision '16-true', 'bf16-true', "
                    f"'transformer-engine', '16-mixed' and 'bf16-mixed'. Using {precision} may cause errors."
                )

        self.HF_model = AutoModelForCausalLM.from_config(self.HF_model_config, **extra_kwargs)

        self.do_demo = do_demo
        if self.do_demo:
            self.forward = self._forward_demo
        else:
            self.forward = self._forward

        if isinstance(gpt_kwargs, DictConfig):
            logger.info("Converting gpt_kwargs from DictConfig to dict.")
            gpt_kwargs = OmegaConf.to_container(gpt_kwargs, resolve=True)
        elif not isinstance(gpt_kwargs, dict):
            logger.warning(f"gpt_kwargs should be a dict or DictConfig, but got {type(gpt_kwargs)}.")

        self.hparams = {
            "gpt_kwargs": gpt_kwargs,
            "precision": precision,
            "do_demo": do_demo,
        }

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

    def _check_parameters(self):
        """Logs a warning about the finiteness of any parameters in the model.

        This is only used for advanced debugging. It does not raise an error because when this mode is
        enabled, typically detect anomaly is on in the lightning trainer, and that gives more information
        about these issues than a generic assertion would.

        Validity checks:
            - The parameters are not nan.
            - The parameters are not inf.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> model.HF_model.gpt_neox.layers[1].attention.query_key_value.bias.shape
            torch.Size([12])
            >>> model.HF_model.gpt_neox.layers[1].attention.query_key_value.bias = torch.nn.Parameter(
            ...     torch.tensor([float("nan"), 0., float("inf"), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            ... )
            >>> with print_warnings():
            ...     model._check_parameters()
            Warning: Parameter HF_model.gpt_neox.layers.1.attention.query_key_value.bias contains 1/12 nan
                values.
            Warning: Parameter HF_model.gpt_neox.layers.1.attention.query_key_value.bias contains 1/12 inf
                values.
        """

        for n, p in self.named_parameters():
            num_nan = _val(torch.isnan(p).sum())
            num_inf = _val(torch.isinf(p).sum())

            if num_nan > 0:
                logger.warning(f"Parameter {n} contains {num_nan}/{p.numel()} nan values.")
            if num_inf > 0:
                logger.warning(f"Parameter {n} contains {num_inf}/{p.numel()} inf values.")

    def _check_outputs(self, loss: torch.FloatTensor, outputs: CausalLMOutputWithPast):
        """Logs a warning if the loss is inf or nan.

        This does not raise an error because when this mode is enabled, typically detect anomaly is on in the
        lightning trainer, and that gives more information about these issues than a generic assertion would.

        Validity checks:
            - The loss is not inf or nan.
            - The logits contain inf or nan values.

        Args:
            loss: The loss tensor.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> fake_output_valid = Mock(logits=torch.FloatTensor([[0.1, 0.2], [0.3, 0.4]]))
            >>> model._check_outputs(torch.tensor(0.4), fake_output_valid) # no errors
            >>> with print_warnings():
            ...     model._check_outputs(torch.tensor(float("inf")), fake_output_valid)
            ...     model._check_outputs(torch.tensor(float("nan")), fake_output_valid)
            Warning: Loss contains inf values.
            Warning: Loss contains nan values.
            >>> fake_output_inf = Mock(logits=torch.FloatTensor([[float("inf"), 0.2], [0.3, 0.4]]))
            >>> fake_output_nan = Mock(logits=torch.FloatTensor([[0.4, float("nan")], [0.3, float("nan")]]))
            >>> with print_warnings():
            ...     model._check_outputs(torch.tensor(0.4), fake_output_inf)
            ...     model._check_outputs(torch.tensor(0.4), fake_output_nan)
            Warning: Logits contains 1/4 inf values.
            Warning: Logits contains 2/4 nan values.
        """

        if _val(torch.isinf(loss).any()):
            logger.warning("Loss contains inf values.")
        if _val(torch.isnan(loss).any()):
            logger.warning("Loss contains nan values.")

        logits = outputs.logits
        inf_count = _val(torch.isinf(logits).sum())
        if inf_count > 0:
            logger.warning(f"Logits contains {inf_count}/{logits.numel()} inf values.")
        nan_count = _val(torch.isnan(logits).sum())
        if nan_count > 0:
            logger.warning(f"Logits contains {nan_count}/{logits.numel()} nan values.")

    def _hf_inputs(self, batch: MEDSTorchBatch) -> dict[str, torch.Tensor]:
        """Converts the MEDSTorchBatch to a dictionary of inputs for the Hugging Face model.

        HF relevant input keys:
            - input_ids: The input sequence of token IDs. Captured in `batch.code`.
            - attention_mask: A mask to avoid attending to padding tokens. See the
              [documentation](https://huggingface.co/docs/transformers/en/model_doc/gpt_neox#transformers.GPTNeoXModel.forward.attention_mask)
              for more details. Should be a tensor of shape `(batch_size, seq_len)` (same as `input_ids`) with
              0s for tokens that are masked and 1s for tokens that are not masked. This means it is given by
              `batch.code != batch.PAD_INDEX` as whenever the code is not a padding token, it should be
              attended to.

        Args:
            batch: The input batch of data.

        Returns:
            A dictionary of inputs for the Hugging Face model.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> batch = Mock(code=torch.LongTensor([[0, 3, 1], [0, 0, 2]]), PAD_INDEX=0)
            >>> model._hf_inputs(batch)
            {'input_ids': tensor([[0, 3, 1],
                                  [0, 0, 2]]),
             'attention_mask': tensor([[False,  True,  True],
                                       [False, False,  True]])}
        """
        return {
            "input_ids": batch.code,
            "attention_mask": (batch.code != batch.PAD_INDEX),
        }

    def _forward_demo(self, batch: MEDSTorchBatch) -> tuple[torch.FloatTensor, CausalLMOutputWithPast]:
        """A demo forward pass that adds more checks and assertions."""

        self._check_inputs(batch)
        self._check_parameters()
        out = self._forward(batch)
        self._check_outputs(*out)

        return out

    def _forward(self, batch: MEDSTorchBatch) -> tuple[torch.FloatTensor, CausalLMOutputWithPast]:
        outputs = self.HF_model(**self._hf_inputs(batch))
        loss = F.cross_entropy(
            outputs.logits[:, :-1].transpose(2, 1), batch.code[:, 1:], ignore_index=batch.PAD_INDEX
        )

        return loss, outputs

    def generate(self, batch: MEDSTorchBatch, do_sample: bool = True, **kwargs) -> torch.Tensor:
        """Generates a sequence of tokens from the model using the HF generation mixin.

        Examples:
            >>> _ = torch.manual_seed(0)
            >>> torch.use_deterministic_algorithms(True)
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 10,
            ...     "vocab_size": dataset_config.vocab_size,
            ... }, precision="32-true")

        This model has a maximum sequence length of 10. If we check, our sample batch has a sequence length of
        9:

            >>> sample_batch.code.shape
            torch.Size([2, 9])

        This means that by default, the model will generate 1 token:

            >>> print(model.generate(sample_batch, do_sample=False))
            tensor([[2],
                    [2]])

        If we create a model with a maximum sequence length of 20, we can generate 11 tokens:

            >>> _ = torch.manual_seed(0)
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 20,
            ...     "vocab_size": dataset_config.vocab_size,
            ... }, precision="32-true")
            >>> print(model.generate(sample_batch, do_sample=False))
            tensor([[ 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
                    [ 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]])

        Note that here, as we've turned sampling off for determinism in these testing algorithms, the
        generated samples are clearly visibly not meaningful. This is completely expected because we are
        working with a model that has just been randomly initialized. What if we try with a model that has
        undergone a (tiny) bit of pre-training? TODO(generation test).
        """

        for_hf = self._hf_inputs(batch)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_seq_len - batch.code.shape[1],
            do_sample=do_sample,
            num_beams=1,  # no beam search
            temperature=1.0,
            pad_token_id=batch.PAD_INDEX,
            bos_token_id=None,
            eos_token_id=self.HF_model.config.eos_token_id,
        )

        output_ids = self.HF_model.generate(
            for_hf.pop("input_ids"),
            generation_config=generation_config,
            **for_hf,
            **kwargs,
        )

        input_seq_len = batch.code.shape[1]

        return output_ids[:, input_seq_len:]

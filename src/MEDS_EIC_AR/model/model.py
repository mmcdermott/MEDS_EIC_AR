import logging
from typing import ClassVar

import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from MEDS_EIC_AR.model.backends import GenerationBackend, HFBackend

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
    """A Llama-style decoder-only transformer for pre-training an autoregressive, "everything-is-code" model.

    Wraps :class:`~transformers.LlamaForCausalLM` so it runs over MEDS-TorchData batches. The rolling-
    generation loop and everything downstream of :attr:`HF_model` are architecture-agnostic, so the base
    is contained to :meth:`__init__` — swapping to a different HF architecture is a one-spot change.

    Args:
        gpt_kwargs: A dictionary of keyword arguments to pass to the underlying HF config constructor (a
            :class:`~transformers.LlamaConfig` today). These can include ``max_position_embeddings``,
            ``vocab_size``, ``hidden_size``, etc. The historical name ``gpt_kwargs`` is retained for
            backwards-compatibility with existing config files.

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
        MEDSTorchBatch(code=tensor([[38,  22, 36,  3, 12, 29, 35,  4, 37],
                                    [38,  22, 36,  2, 21, 25, 35,  4, 37]]),
                       ...)

    We run over the batch in the normal way, which internally calls the `forward` method of the model:

        >>> loss, outputs = model(sample_batch)
        >>> print(loss)
        tensor(3.6543, dtype=torch.float16, grad_fn=<NllLoss2DBackward0>)
        >>> print(f"Outputs have keys: {', '.join(outputs.keys())}")
        Outputs have keys: logits, past_key_values
        >>> print(f"Logits shape: {outputs.logits.shape}")
        Logits shape: torch.Size([2, 9, 39])
        >>> print(outputs.logits)
        tensor([[[ 2.5539e-03, ..., -3.0045e-02], ..., [-8.3694e-03, ...,  3.0487e-02]],
        <BLANKLINE>
                [[ 2.5539e-03, ..., -3.0045e-02], ..., [-9.0561e-03, ...,  3.5919e-02]]],
               dtype=torch.float16,
               grad_fn=<UnsafeViewBackward0>)

    The models parameters can be accessed in the normal way. The first named parameter is the token
    embedding matrix:

        >>> sample_param_name, sample_param = next(iter(model.named_parameters()))
        >>> print(f"{sample_param_name} ({sample_param.shape}): {sample_param}")
        HF_model.model.embed_tokens.weight (torch.Size([39, 4])): Parameter containing:
        tensor([[ 0.0069,  0.0068, -0.0367,  0.0099], ..., [-0.0085, -0.0223, -0.0185,  0.0032]],
               dtype=torch.float16,
               requires_grad=True)

    Let's validate that they have gradients that can be realized via `.backward()` as normal:

        >>> print(f"Sample parameter grad?: {sample_param.grad}")
        Sample parameter grad?: None
        >>> loss.backward()
        >>> print(f"Sample parameter grad?: {sample_param.grad}")
        Sample parameter grad?:
        tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
                ...,
                [-0.1140, -0.0175,  0.0645, -0.0845]],
               dtype=torch.float16)

    With a single backward pass, we should not get any infinite gradients:

        >>> for name, param in model.named_parameters():
        ...     if param.grad is not None:
        ...         if not _val(torch.isfinite(param.grad).all()):
        ...             raise ValueError(f"Gradient for {name} is not finite.")

    Model errors are raised if we pass invalid kwargs to the constructor. The architecture name in the
    error message is derived from the active HF config's ``model_type`` so it stays accurate if we ever
    swap the base architecture again:

        >>> Model({"foobar": 2})
        Traceback (most recent call last):
            ...
        ValueError: Config for HF model llama does not have attribute foobar

    Post-override snap: ``num_key_value_heads`` and ``head_dim`` are recomputed from the caller's
    ``hidden_size`` / ``num_attention_heads`` unless the caller set them explicitly. The default is
    plain MHA (``num_key_value_heads == num_attention_heads``) — matching what vanilla
    :class:`~transformers.LlamaConfig` does when either field is left unset:

        >>> tiny = Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 2,
        ...     "hidden_size": 4,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": 40,
        ... }, precision="32-true")
        >>> tiny.HF_model_config.num_key_value_heads
        2
        >>> tiny.HF_model_config.head_dim
        2

    A caller who wants GQA sets ``num_key_value_heads`` explicitly:

        >>> gqa = Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 4,
        ...     "num_key_value_heads": 2,
        ...     "hidden_size": 8,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": 40,
        ... }, precision="32-true")
        >>> gqa.HF_model_config.num_key_value_heads
        2
        >>> gqa.HF_model_config.head_dim
        2

    And a non-divisible ``hidden_size`` / ``num_attention_heads`` combination is rejected at
    construction time rather than producing a silently-wrong ``head_dim`` that only surfaces as a
    cryptic shape error on the first forward pass:

        >>> Model({
        ...     "num_hidden_layers": 2,
        ...     "num_attention_heads": 3,
        ...     "hidden_size": 4,
        ...     "max_position_embeddings": 10,
        ...     "vocab_size": 40,
        ... })
        Traceback (most recent call last):
            ...
        ValueError: hidden_size (4) must be divisible by num_attention_heads (3) to derive ...
    """

    HF_model_config: LlamaConfig
    HF_model: LlamaForCausalLM
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

    _RESERVED_ROLLING_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"generation_config", "eos_token_id", "pad_token_id", "bos_token_id", "max_new_tokens"}
    )

    def __init__(self, gpt_kwargs: dict | DictConfig, precision: str = "32-true", do_demo: bool = False):
        super().__init__()

        self.HF_model_config: LlamaConfig = LlamaConfig()

        for key, val in gpt_kwargs.items():
            # ``attention_head_dim`` is consumed by the YAML ``int_prod`` resolver to derive
            # ``hidden_size`` (``hidden_size = num_attention_heads * attention_head_dim``). The
            # config expresses the product form so HPO search over ``num_attention_heads`` and
            # ``attention_head_dim`` independently can never produce a non-divisible
            # ``hidden_size``. It isn't a real LlamaConfig attribute — skip it on its way in.
            if key == "attention_head_dim":
                continue
            if not hasattr(self.HF_model_config, key):
                raise ValueError(
                    f"Config for HF model {self.HF_model_config.model_type} does not have attribute {key}"
                )
            setattr(self.HF_model_config, key, val)

        # ``LlamaConfig()``'s default values for ``num_key_value_heads`` and ``head_dim`` come from
        # the vanilla 7B config — they aren't recomputed from the ``hidden_size`` /
        # ``num_attention_heads`` the caller just overrode. Snap them back to the values implied
        # by the overrides unless the caller also set them explicitly. Plain MHA
        # (``num_key_value_heads == num_attention_heads``) and
        # ``head_dim == hidden_size // num_attention_heads`` are what vanilla Llama uses when the
        # fields are left unset at config-construction time; this just preserves that after our
        # post-hoc attribute mutation.
        if "num_key_value_heads" not in gpt_kwargs:
            self.HF_model_config.num_key_value_heads = self.HF_model_config.num_attention_heads
        if "head_dim" not in gpt_kwargs:
            hidden = self.HF_model_config.hidden_size
            heads = self.HF_model_config.num_attention_heads
            if hidden % heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden}) must be divisible by num_attention_heads ({heads}) to "
                    "derive ``head_dim`` implicitly. Pass ``head_dim`` explicitly in gpt_kwargs, or "
                    "adjust ``hidden_size`` / ``num_attention_heads`` so the division is exact. "
                    "Configs built via the ``attention_head_dim`` resolver at "
                    "``configs/lightning_module/model/default.yaml`` cannot hit this — the product "
                    "form makes divisibility impossible-by-construction."
                )
            self.HF_model_config.head_dim = hidden // heads

        extra_kwargs = {"dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}

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

        # Generation backend abstraction (issue #88). The default HF backend is a thin wrapper
        # that produces byte-identical behavior to the pre-abstraction direct ``HF_model.generate``
        # call. Alternative backends (SGLang, …) can be swapped in via :meth:`set_backend`
        # without touching the rolling loop.
        self._backend: GenerationBackend = HFBackend(self.HF_model)

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
    def backend(self) -> GenerationBackend:
        """The active generation backend used by :meth:`_generate_chunk`."""
        return self._backend

    def set_backend(self, backend: GenerationBackend) -> None:
        """Swap in an alternative generation backend.

        Intended for the SGLang adapter (issue #88) and for injecting deterministic fakes in
        tests. No-op on its own — rolling-loop bookkeeping is backend-agnostic, so behavior is
        fully determined by the backend's ``generate_chunk`` implementation.
        """
        self._backend = backend

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

        _batch_size, seq_len = code.shape

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
            We corrupt a ``RMSNorm`` weight to exercise the warning path. The parameter choice
            doesn't matter; any tensor with known ``numel`` works.

            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> model.HF_model.model.layers[1].input_layernorm.weight.shape
            torch.Size([4])
            >>> model.HF_model.model.layers[1].input_layernorm.weight = torch.nn.Parameter(
            ...     torch.tensor([float("nan"), 0., float("inf"), 0.])
            ... )
            >>> with print_warnings():
            ...     model._check_parameters()
            Warning: Parameter HF_model.model.layers.1.input_layernorm.weight contains 1/4 nan values.
            Warning: Parameter HF_model.model.layers.1.input_layernorm.weight contains 1/4 inf values.
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
              [documentation](https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaModel.forward.attention_mask)
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

    def generate(
        self,
        batch: MEDSTorchBatch,
        do_sample: bool = True,
        max_new_tokens: int | None = None,
        rolling_context_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generates a sequence of tokens from the model.

        When ``max_new_tokens`` is ``None`` (the default), this performs a single call to the Hugging Face
        generation mixin, bounded by ``max_seq_len - input_len`` — the model can only emit as many tokens as
        its remaining context window. This is the legacy, "generate to the end of the input window" path.

        When ``max_new_tokens`` is an integer, this switches to a rolling sliding-window loop that can emit up
        to ``max_new_tokens`` tokens in total, even if that exceeds the model's positional embedding limit.
        The loop repeatedly calls the underlying HF generate with the right-aligned tail of
        ``(original_input ++ previously_generated)`` (truncated to ``rolling_context_size``) until either
        every sample has emitted the EOS token (``TIMELINE//END``) or ``max_new_tokens`` new tokens have been
        produced in total. Samples that have already emitted EOS are zeroed into padding on subsequent chunks.
        See :meth:`_rolling_generate` for details.

        Args:
            batch: The input batch of data. Only ``batch.code`` and ``batch.PAD_INDEX`` are read.
            do_sample: Whether HF sampling is enabled (``True``) or greedy (``False``).
            max_new_tokens: Total new-token budget across all chunks. ``None`` selects the legacy single-call
                path.
            rolling_context_size: Size of the sliding context window fed to the model on each chunk. Only used
                when ``max_new_tokens`` is set. Defaults to ``max_seq_len - 1`` so every chunk always has room
                for at least one new token.
            **kwargs: Forwarded to ``HF_model.generate``.

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
            tensor([[38],
                    [38]])

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
            tensor([[38,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8],
                    [38,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8]])

        Setting ``max_new_tokens`` explicitly enables the sliding-window path, which can emit more tokens than
        the model's context window would normally allow. Here the model has ``max_position_embeddings=10`` but
        we request 15 new tokens. Rolling generation requires a valid ``eos_token_id`` that differs from
        ``batch.PAD_INDEX``, which in a real run is set by the ``MEICAR_pretrain`` CLI to the index of
        ``TIMELINE//END``; we set it manually here because the randomly-initialized test model has no
        pretraining and defaults to ``eos_token_id=0``:

            >>> _ = torch.manual_seed(0)
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 10,
            ...     "vocab_size": dataset_config.vocab_size,
            ... }, precision="32-true")
            >>> model.HF_model.config.eos_token_id = 37  # TIMELINE//END for the test vocab
            >>> model.generate(sample_batch, do_sample=False, max_new_tokens=15)
            tensor([[38,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8],
                    [38,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8]])

        Note that here, as we've turned sampling off for determinism in these testing algorithms, the
        generated samples are clearly visibly not meaningful — a randomly initialized model greedy-decodes
        to a small cycle of high-probability tokens and repeats it. This is completely expected. What if
        we try with a model that has undergone a (tiny) bit of pre-training? TODO(generation test).

        Note on HF library support: HF Transformers does not currently provide a built-in way to generate
        more tokens than the model's ``max_position_embeddings`` for architectures without native sliding-
        window attention. ``LlamaForCausalLM``, which this model wraps, is one such architecture (plain
        Llama is MHA + full attention; the sliding-window variants — Mistral, Gemma2, Llama4 — are
        separate architectures). The earlier ``SinkCache`` / StreamingLLM approach was removed from
        recent releases; its replacement lives inside the model implementations themselves and only
        applies to architectures that advertise sliding-window attention. That's why we maintain our
        own rolling loop here — see :meth:`_rolling_generate` for the trade-offs, including the
        positional-embedding correctness consideration when the window slides.
        """

        if max_new_tokens is not None:
            return self._rolling_generate(
                batch,
                max_new_tokens=max_new_tokens,
                rolling_context_size=rolling_context_size,
                do_sample=do_sample,
                **kwargs,
            )

        for_hf = self._hf_inputs(batch)
        input_ids = for_hf.pop("input_ids")
        return self._generate_chunk(
            input_ids,
            attention_mask=for_hf.get("attention_mask"),
            max_new_tokens=self.max_seq_len - batch.code.shape[1],
            pad_id=batch.PAD_INDEX,
            do_sample=do_sample,
            **kwargs,
        )

    def _rolling_generate(
        self,
        batch: MEDSTorchBatch,
        max_new_tokens: int,
        rolling_context_size: int | None,
        do_sample: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Sliding-window generation loop that can exceed the model's context window.

        The loop preallocates a single ``[B, L_in + max_new_tokens]`` buffer ``sequence_so_far`` that
        starts identical to ``input_ids`` and grows as new tokens are written into its tail. On each
        iteration the next chunk's prompt is the right-aligned ``ctx_size``-wide slice of
        ``sequence_so_far[:, :n_total]``, where ``n_total = L_in + n_generated`` — no separate "input"
        and "generated" tensors, no three-case prompt construction, no per-step ``torch.cat``. At the
        end we simply slice off the input prefix and return ``sequence_so_far[:, L_in : L_in + n_generated]``.
        Memory movement for buffer growth is therefore O(T·ctx); total generation compute is still
        dominated by the repeated ``HF_model.generate`` forward passes (roughly O(#chunks · ctx²) for
        full attention).

        HF ``generate`` handles in-chunk EOS (``TIMELINE//END``) naturally: rows that hit EOS are
        padded with ``PAD_INDEX`` from that point on. Across chunks we track a ``finished`` mask and
        overwrite tokens produced for already-finished rows with pad before writing them, so those
        rows don't "resurrect" when the next chunk's prompt re-exposes their EOS token. The loop
        terminates as soon as every row is finished or the new-token budget is exhausted.

        **Why we maintain our own loop.** HF Transformers mainline currently exposes no first-class
        primitive for generating past a full-attention model's ``max_position_embeddings``. ``SinkCache``
        was removed in 4.53.0; the sliding-window-attention replacements live inside specific architectures
        (Mistral, Gemma2, Llama4) that handle the sliding in-kernel, and don't apply to plain
        ``LlamaForCausalLM``, which this repo's :class:`Model` wraps. The remaining HF caches
        (``DynamicCache``, ``StaticCache``,
        ``QuantizedCache``, offloaded variants) only manage memory — none of them let the model see
        positions beyond ``max_position_embeddings``. A community port
        (`transformers-community/sink_cache <https://huggingface.co/transformers-community/sink_cache>`_)
        preserves the old SinkCache API and is a candidate for a future efficiency pass on this loop; it is
        not a correctness prerequisite for what we do here (see below).

        **Correctness framing.** This loop is *cache-less* across chunks: every iteration discards the KV
        cache and re-calls ``HF_model.generate(...)`` on a fresh prompt. HF builds fresh ``position_ids``
        starting at 0 and applies RoPE rotations at those fresh positions, so every chunk is just "here is
        a sequence of length ≤ ``max_seq_len``, continue it" — a valid, in-distribution prefix by the model's
        standards. There is no RoPE violation, no stale cached rotations, no attention geometry outside the
        trained regime. The loop inherits exactly the error profile of running the model on a short input
        from scratch, chunk after chunk.

        What the loop cannot avoid is the generic finite-context information-loss problem: once a token
        scrolls off the left edge, the model can no longer attend to it. This is the same limitation any
        fixed-window model has when its input exceeds its window; rolling generation makes it visible rather
        than introducing a new error class. The secondary concern is ordinary compounding-error, which can
        manifest in loop-specific ways near sliding boundaries because the "start of the visible prefix"
        changes content each chunk — an empirical/behavioral concern, not a mathematical one. Callers who
        want to avoid even that tradeoff can leave ``max_new_tokens=None`` and stay on the single-chunk path.

        A cheap future experiment: prepend the first K tokens of the original input to every chunk's prompt
        to approximate StreamingLLM's attention-sink pinning without any library dependency. Not needed for
        correctness; potentially useful as a quality knob on real pretrained checkpoints.

        **Backend evolution.** Under the SGLang backend proposed in #88, only the inner
        :meth:`_generate_chunk` call site inside this loop is swapped for a backend-level
        ``generate_chunk(...)`` invocation. The sliding-window bookkeeping — the ``sequence_so_far``
        buffer, the slice-based prompt construction, the ``finished`` mask — stays here because it is
        the right place to own cross-chunk state (including the time-budget stopping criterion from
        #82 and the REACH logits-processor state described in the #87 SCOPE/REACH comment thread).

        Args:
            batch: Input batch. Only ``batch.code`` and ``batch.PAD_INDEX`` are read.
            max_new_tokens: Total new-token budget across all chunks. Must be positive.
            rolling_context_size: Per-chunk context window. ``None`` defaults to ``max_seq_len - 1``, the
                largest value that still leaves room for at least one new token per chunk once the window
                saturates.
            do_sample: Whether HF sampling is enabled.
            **kwargs: Forwarded to ``HF_model.generate``.

        Returns:
            A ``[B, L]`` tensor of newly generated tokens, where ``L <= max_new_tokens``. Each row
            has ``PAD_INDEX`` from its first EOS onwards (HF pads within each chunk, and the
            cross-chunk ``finished`` mask keeps already-finished rows padded in subsequent chunks).

        Raises:
            ValueError: If ``max_new_tokens`` or ``rolling_context_size`` are non-positive; if the
                model's ``eos_token_id`` is unset or collides with ``batch.PAD_INDEX``; or if
                ``kwargs`` contains any of the HF ``generate`` keys that this loop manages internally
                (``generation_config``, ``eos_token_id``, ``pad_token_id``, ``bos_token_id``,
                ``max_new_tokens``). Those would override the values we bake into the per-chunk
                ``GenerationConfig`` and desynchronize HF's in-chunk stopping from the cross-chunk
                ``finished`` mask, silently producing incorrect outputs.

        Examples:
            We can build a tiny deterministic model to exercise the loop. With
            ``max_position_embeddings=5`` and ``max_new_tokens=12``, generation must roll across at least
            three chunks because each chunk can only produce at most ``5 - 1 = 4`` new tokens once the
            sliding window is full:

            >>> _ = torch.manual_seed(0)
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 5,
            ...     "vocab_size": dataset_config.vocab_size,
            ... }, precision="32-true")
            >>> model.HF_model.config.eos_token_id = 37  # TIMELINE//END for the test vocab
            >>> batch = Mock(
            ...     code=torch.LongTensor([[38, 22, 36], [38, 22, 36]]),
            ...     PAD_INDEX=0,
            ...     mode="SM",
            ... )

            We wrap ``HF_model.generate`` with a spy so the doctest can also assert that the rolling loop
            actually iterated across multiple chunks — not just that it produced a plausible-looking output
            tensor from a single inner call:

            >>> _real_generate = model.HF_model.generate
            >>> model.HF_model.generate = MagicMock(wraps=_real_generate)
            >>> model._rolling_generate(
            ...     batch, max_new_tokens=12, rolling_context_size=None, do_sample=False
            ... )
            tensor([[ 3, 38,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8],
                    [ 3, 38,  7,  8, 28, 15,  7,  8, 28, 15,  7,  8]])
            >>> model.HF_model.generate.call_count >= 3
            True
            >>> model.HF_model.generate = _real_generate

            The output is a deterministic (seeded, greedy) smoke test: the randomly-initialized model
            never samples the EOS token (37), so no row is truncated. Post-EOS padding, when it fires,
            is handled directly by HF ``generate`` inside each chunk. The ``call_count >= 3`` check is
            the real integration signal: with ``max_position_embeddings=5`` and a 3-token prompt, a
            single inner ``HF_model.generate`` call can emit at most ``5 - 3 = 2`` new tokens on the
            first chunk and
            ``5 - (5 - 1) = 1`` new token on each subsequent chunk once the sliding window saturates, so
            reaching 12 total new tokens requires at least ``1 + 10 = 11`` chunks — well above the
            ``>= 3`` floor we assert, and impossible via a single-chunk fall-through.

            If we instead choose a ``max_new_tokens`` that is smaller than the model's one-shot capacity,
            we still get exactly that many tokens back — the rolling path degenerates into a single chunk
            with a tighter ``max_new_tokens`` cap:

            >>> _ = torch.manual_seed(0)
            >>> model._rolling_generate(
            ...     batch, max_new_tokens=2, rolling_context_size=None, do_sample=False
            ... )
            tensor([[ 3, 38],
                    [ 3, 38]])

        Validation errors:

            >>> model._rolling_generate(
            ...     batch, max_new_tokens=0, rolling_context_size=None, do_sample=False
            ... )
            Traceback (most recent call last):
                ...
            ValueError: max_new_tokens must be positive; got 0.
            >>> model._rolling_generate(
            ...     batch, max_new_tokens=5, rolling_context_size=0, do_sample=False
            ... )
            Traceback (most recent call last):
                ...
            ValueError: rolling_context_size must be positive; got 0.

            >>> model.HF_model.config.eos_token_id = None
            >>> model._rolling_generate(
            ...     batch, max_new_tokens=5, rolling_context_size=None, do_sample=False
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Rolling generation requires the model's eos_token_id to be set (got None). ...
            >>> model.HF_model.config.eos_token_id = 0  # collides with PAD_INDEX
            >>> model._rolling_generate(
            ...     batch, max_new_tokens=5, rolling_context_size=None, do_sample=False
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Rolling generation requires eos_token_id (0) to differ from batch.PAD_INDEX (0). ...

            Passing reserved HF ``generate`` kwargs through ``**kwargs`` is rejected, because they
            would override the values this loop bakes into each chunk's ``GenerationConfig`` and
            desynchronize HF's in-chunk stopping from our cross-chunk state:

            >>> model.HF_model.config.eos_token_id = 37
            >>> model._rolling_generate(
            ...     batch, max_new_tokens=5, rolling_context_size=None, do_sample=False,
            ...     eos_token_id=99,
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Rolling generation manages these HF generate kwargs internally ...
        """

        reserved = Model._RESERVED_ROLLING_KWARGS & kwargs.keys()
        if reserved:
            raise ValueError(
                "Rolling generation manages these HF generate kwargs internally and cannot "
                f"accept them via **kwargs: {sorted(reserved)}. Control stopping/padding via "
                "Model.HF_model.config.eos_token_id and batch.PAD_INDEX, and control the "
                "new-token budget via Model.generate(max_new_tokens=...) / rolling_context_size."
            )

        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive; got {max_new_tokens}.")

        if rolling_context_size is None:
            ctx_size = self.max_seq_len - 1
        else:
            if rolling_context_size <= 0:
                raise ValueError(f"rolling_context_size must be positive; got {rolling_context_size}.")
            ctx_size = min(rolling_context_size, self.max_seq_len - 1)

        input_ids = batch.code
        pad_id = batch.PAD_INDEX
        eos_id = self.HF_model.config.eos_token_id

        if eos_id is None:
            raise ValueError(
                "Rolling generation requires the model's eos_token_id to be set (got None). "
                "Instantiate the model with eos_token_id=get_timeline_end_token_idx(dataset_config), "
                "as is done in the MEICAR_pretrain entry point."
            )
        if eos_id == pad_id:
            raise ValueError(
                f"Rolling generation requires eos_token_id ({eos_id}) to differ from "
                f"batch.PAD_INDEX ({pad_id}). The finished-mask and post-EOS truncation would "
                "otherwise collapse onto padding."
            )

        batch_size, input_len = input_ids.size()
        device = input_ids.device

        # Preallocate a single ``sequence_so_far`` buffer that starts identical to ``input_ids`` and
        # has room for ``max_new_tokens`` more tokens appended at the right. Every iteration reads a
        # right-aligned ``ctx_size``-wide slice of ``sequence_so_far[:, :n_total]`` as the prompt and
        # writes its new tokens into ``sequence_so_far[:, n_total : n_total + new_len]``. Pad-init so
        # any slice is a semantically valid token tensor even before it's written to.
        sequence_so_far = torch.full(
            (batch_size, input_len + max_new_tokens), pad_id, dtype=input_ids.dtype, device=device
        )
        sequence_so_far[:, :input_len] = input_ids
        n_total = input_len  # current length of the logical sequence in ``sequence_so_far``
        n_generated = 0
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        while n_generated < max_new_tokens and not bool(finished.all()):
            window_start = max(0, n_total - ctx_size)
            prompt = sequence_so_far[:, window_start:n_total]
            prompt_len = prompt.size(1)
            prompt_mask = prompt != pad_id

            # ``prompt_len <= ctx_size <= max_seq_len - 1``, so ``max_seq_len - prompt_len >= 1`` and
            # the per-chunk budget is always positive. The ``min`` only kicks in on the last iteration
            # when the remaining budget is smaller than the per-call cap.
            chunk_budget = min(max_new_tokens - n_generated, self.max_seq_len - prompt_len)
            new_tokens = self._generate_chunk(
                prompt,
                attention_mask=prompt_mask,
                max_new_tokens=chunk_budget,
                pad_id=pad_id,
                do_sample=do_sample,
                **kwargs,
            )
            new_len = new_tokens.size(1)

            # Mask out tokens for already-finished rows so they stay padded in the buffer. HF already
            # pads post-EOS within each chunk via ``pad_token_id`` during generation, so within-chunk
            # post-EOS positions are already correct; this mask only keeps rows that finished in
            # *earlier* chunks from resurrecting on the next iteration (because the new prompt
            # includes the EOS token in its visible tail, and HF would happily keep going).
            if bool(finished.any()):
                new_tokens = new_tokens.masked_fill(finished.unsqueeze(1), pad_id)

            sequence_so_far[:, n_total : n_total + new_len] = new_tokens
            n_total += new_len
            n_generated += new_len

            hit_eos = (new_tokens == eos_id).any(dim=1)
            finished = finished | hit_eos

        # Drop the original input prefix and return only the generated tail. No explicit post-EOS
        # truncation is needed: HF's ``generate`` pads with ``pad_id`` after EOS within each chunk,
        # and the ``finished`` mask above keeps already-finished rows padded across chunks, so every
        # row in the returned slice already has ``pad_id`` from its first EOS onwards.
        return sequence_so_far[:, input_len : input_len + n_generated]

    def _generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        pad_id: int,
        do_sample: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Run one generate pass via :attr:`backend` and return only the newly generated tokens.

        Shared between :meth:`generate`'s single-chunk path and :meth:`_rolling_generate`'s per-chunk
        loop so both paths build the ``GenerationConfig`` the same way and apply the same prompt-
        stripping convention. The actual engine call is delegated to :attr:`backend` (HF by default;
        see issue #88). The :class:`~MEDS_EIC_AR.model.backends.GenerationBackend` contract requires
        backends to return chunk outputs such that, within a single call, positions after the first
        EOS per row are already filled with ``pad_id`` — so neither caller here has to do post-EOS
        truncation of the per-chunk return value. :class:`~MEDS_EIC_AR.model.backends.HFBackend`
        satisfies this for free because HF's ``generate`` pads post-EOS natively (verified
        empirically on ``LlamaForCausalLM``); non-HF backends must honor the invariant
        explicitly. ``eos_token_id`` is read from ``self.HF_model.config.eos_token_id`` rather than
        passed in, since both callers always want the model's configured EOS.

        Args:
            input_ids: ``[B, L_in]`` tensor of prompt tokens.
            attention_mask: Optional ``[B, L_in]`` attention mask (``True`` for real tokens).
            max_new_tokens: Per-call budget passed into ``GenerationConfig``.
            pad_id: Pad token id. Comes from the ``batch`` at call time, not the model config.
            do_sample: Whether to sample (``True``) or greedy-decode (``False``).
            **kwargs: Forwarded to the active backend's ``generate_chunk``.

        Returns:
            A ``[B, new_len]`` tensor of newly generated tokens, sliced off the prompt.
        """
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            temperature=1.0,
            pad_token_id=pad_id,
            bos_token_id=None,
            eos_token_id=self.HF_model.config.eos_token_id,
        )
        return self._backend.generate_chunk(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )

import logging

import torch
from meds_torchdata import MEDSTorchBatch
from omegaconf import ListConfig
from torchmetrics import Metric, MetricCollection
from torchmetrics.text import Perplexity

logger = logging.getLogger(__name__)


class TopKMulticlassAccuracy(Metric):
    """Computes the fraction of samples that yield a correct prediction within their top-k class logits.

    The torchmetrics codebase Accuracy metric does not perform what I expect it to in top-k and multi-class
    settings. This metric behaves as expected.

    See https://github.com/Lightning-AI/torchmetrics/issues/3068 for tracking.

    Args:
        top_k: A prediction for a given sequence event is correct if the true label is among the top `top_k`
            predicted logits.
        ignore_index: The index to ignore when computing the accuracy. This is useful for ignoring padding
            indices. For this metric, it must be set.

    Returns:
        A (scalar) tensor with the accuracy of the top-k predictions. If there are no valid predictions (e.g.,
        all labels are padding), returns 0.0.

    Raises:
        ValueError: If the logits and target tensors do not have the same shape in the batch (0) and sequence
            (1) dimensions.

    Examples:
        >>> code = torch.LongTensor([[3, 2], [1, 0]])
        >>> logits = torch.FloatTensor(
        ...     [
        ...         [[0.0, 0.1, 0.5, 0.4], [0.0, 0.2, 0.7, 0.1]], # Prediction orders: 2, 3, 1; 2, 1, 3
        ...         [[0.0, 0.4, 0.3, 0.3], [1.0, 0.0, 0.0, 0.0]], # Prediction orders: 1, 2 & 3; padding
        ...     ]
        ... )
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(1.)
        >>> TopKMulticlassAccuracy(top_k=1, ignore_index=0)(logits, code)
        tensor(0.6667)
        >>> code = torch.LongTensor([[1, 1], [1, 0]])
        >>> TopKMulticlassAccuracy(top_k=1, ignore_index=0)(logits, code)
        tensor(0.3333)
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(0.6667)
        >>> TopKMulticlassAccuracy(top_k=3, ignore_index=0)(logits, code)
        tensor(1.)
        >>> code = torch.LongTensor([[1, 0], [0, 0]])
        >>> TopKMulticlassAccuracy(top_k=3, ignore_index=0)(logits, code)
        tensor(1.)
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(0.)

    If all labels are ignored, the accuracy is 0.0:

        >>> code = torch.LongTensor([[0, 0], [0, 0]])
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(0.)

    Errors are raised if the shapes are misaligned:

        >>> code = torch.LongTensor([[1, 1, 0], [0, 3, 0]])
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        Traceback (most recent call last):
            ...
        ValueError: logits and target must have the same shape in the batch (0) and sequence (1) dimensions.
            Got torch.Size([2, 2, 4]) and torch.Size([2, 3])
    """

    top_k: int
    ignore_index: int

    def __init__(self, top_k: int, ignore_index: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.top_k = top_k
        self.ignore_index = ignore_index

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.FloatTensor, target: torch.LongTensor) -> None:
        batch, seq_len = target.shape
        if logits.shape[0] != batch or logits.shape[1] != seq_len:
            raise ValueError(
                "logits and target must have the same shape in the batch (0) and sequence (1) dimensions. "
                f"Got {logits.shape} and {target.shape}"
            )

        topk = torch.topk(logits, k=self.top_k, dim=2).indices  # batch x seq_len x top_k
        correct = (topk == target.unsqueeze(2)).any(dim=2)  # batch x seq_len

        keep_mask = target != self.ignore_index

        self.correct += correct[keep_mask].sum()
        self.total += keep_mask.sum()

    def compute(self) -> torch.FloatTensor:
        safe_denom = torch.where(self.total > 0, self.total, torch.ones_like(self.total))

        return torch.where(self.total > 0, self.correct.float() / safe_denom, torch.tensor(0.0))


class NextCodeMetrics(Metric):
    """A `torchmetrics` Metric for next code prediction in Autoregressive "Everything-is-code" models.

    This module is largely a simple wrapper around `torchmetrics.MetricCollection` to enable configuration and
    code isolation and to seamlessly slice the predictions and tokens to the right shapes.

    Supported metrics:
      - Top-$k$ accuracy
      - Perplexity

    Supported Vocabulary Subdivisions:
      - All codes

    Attributes:
        accuracies: The top-$k$ accuracy metrics contained in this metric collection.
        perplexity: The perplexity metric.

    Examples:
        >>> M = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=4)

    To show it in use, we'll need some codes (targets) and logits (predictions):

        >>> code = torch.LongTensor([[1, 3, 2], [2, 1, 0]])
        >>> logits = torch.FloatTensor([
        ...    [
        ...        [0.0, 0.1, 0.5, 0.4], # Label is 3; Prediction order is 2, 3, 1
        ...        [0.0, 0.2, 0.7, 0.1], # Label is 2; Prediction order is 2, 1, 3
        ...        [0.0, 1.0, 0.0, 0.0], # No label; Should be dropped.
        ...    ], [
        ...        [0.0, 0.4, 0.3, 0.3], # Label is 1; Prediction order is 1, 2 & 3
        ...        [1.0, 0.0, 0.0, 0.0], # Padding label; Should be ignored.
        ...        [0.0, 0.0, 2.0, 0.0], # No label; Should be dropped.
        ...    ]
        ... ])

    We'll make a mock batch as this metric will just use the `code` attribute:

        >>> batch = Mock(spec=MEDSTorchBatch)
        >>> batch.code = code

    Then, we can update and compute the metric values:

        >>> M(logits, batch)
        {'Accuracy/top_1': tensor(0.6667), 'Accuracy/top_2': tensor(1.), 'Accuracy/top_3': tensor(1.),
         'perplexity': tensor(3.1896)}

    You can also run on a single `top_k`:

        >>> M = NextCodeMetrics(top_k=1, vocab_size=4)

    If `top_k` is not an int or a list of ints, an error is raised:

        >>> M = NextCodeMetrics(top_k=[1, "foo"], vocab_size=4)
        Traceback (most recent call last):
            ...
        ValueError: Invalid type for top_k. Want list[int] | int, got <class 'list'> ([1, 'foo']).

    If the max `top_k` is greater than the vocab size, a warning is logged and the `top_k` is filtered to only
    those valid `top_k` values:

        >>> with print_warnings():
        ...     M = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=3)
        Warning: Top-k accuracy requested for k (3 >= vocab_size (3). This is not a valid metric. Filtering to
        only requested k < 3.
        >>> sorted(M.accuracies.keys())
        ['Accuracy/top_1', 'Accuracy/top_2']

    If no valid `top_k` is requested, a warning is logged and the `top_k` is set to 1:

        >>> with print_warnings():
        ...     M = NextCodeMetrics(top_k=[], vocab_size=2)
        Warning: No valid top-k accuracy requested. Adding top-k of 1.
        >>> sorted(M.accuracies.keys())
        ['Accuracy/top_1']
    """

    def __init__(self, top_k: list[int] | int, vocab_size: int, ignore_index: int = 0, **base_metric_kwargs):
        super().__init__(**base_metric_kwargs)

        match top_k:
            case int():
                top_k = [top_k]
            case list() | ListConfig() if all(isinstance(k, int) for k in top_k):
                pass
            case _:
                raise ValueError(
                    f"Invalid type for top_k. Want list[int] | int, got {type(top_k)} ({top_k})."
                )

        if top_k and (max(top_k) >= vocab_size):
            logger.warning(
                f"Top-k accuracy requested for k ({max(top_k)} >= vocab_size ({vocab_size}). "
                f"This is not a valid metric. Filtering to only requested k < {vocab_size}."
            )
            top_k = [k for k in top_k if k < vocab_size]

        if not top_k:
            logger.warning("No valid top-k accuracy requested. Adding top-k of 1.")
            top_k = [1]

        self.accuracies = MetricCollection(
            {f"Accuracy/top_{k}": TopKMulticlassAccuracy(top_k=k, ignore_index=ignore_index) for k in top_k}
        )
        self.perplexity = Perplexity(ignore_index=ignore_index)

    def update(self, logits: torch.Tensor, batch: MEDSTorchBatch):
        """Update the metric with the current batch and logits, sliced to match targets and predictions.

        Args:
            logits: The logits from the model, of shape (batch_size, sequence_length, vocab_size).
            batch: The MEDSTorchBatch containing the input codes at attribute `.code`.  Note that the `code`
                and logits are aligned such that the given code inputs are at the same position as the logits
                produced at that input -- so the logits need to be shifted to align with their prediction
                targets.
        """

        logits = logits[:, :-1]
        targets = batch.code[:, 1:]

        self.perplexity.update(logits, targets)
        self.accuracies.update(logits, targets)

    def compute(self):
        return {**self.accuracies.compute(), "perplexity": self.perplexity.compute()}

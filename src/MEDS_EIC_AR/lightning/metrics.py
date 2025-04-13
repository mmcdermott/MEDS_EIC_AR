import torch
from meds_torchdata import MEDSTorchBatch
from torchmetrics import Accuracy, Metric, MetricCollection
from torchmetrics.text import Perplexity


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

        >>> M.update(logits, batch)
        >>> M.compute()
        {'Accuracy/top_1': tensor(0.6667), 'Accuracy/top_2': tensor(1.0), 'Accuracy/top_3': tensor(1.0),
         'perplexity': tensor(???)}
    """

    def __init__(self, top_k: list[int] | int, vocab_size: int, ignore_index: int = 0, **base_metric_kwargs):
        super().__init__(**base_metric_kwargs)

        match top_k:
            case int():
                top_k = [top_k]
            case list() if all(isinstance(k, int) for k in top_k):
                pass
            case _:
                raise ValueError(f"Invalid type for top_k. Want list[int] | int, got {type(top_k)}")

        acc_kwargs = {
            "task": "multiclass",
            "ignore_index": ignore_index,
            "num_classes": vocab_size,
            "multidim_average": "global",
            "average": "micro",
        }

        self.accuracies = MetricCollection(
            {f"Accuracy/top_{k}": Accuracy(top_k=k, **acc_kwargs) for k in top_k}
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

        # Accuracy metrics expect the logits to be of shape (batch_size, vocab_size, ...), not
        # (batch_size, ..., vocab_size)

        self.accuracies.update(logits.transpose(2, 1), targets)

    def compute(self):
        return {**self.accuracies.compute(), "perplexity": self.perplexity.compute()}

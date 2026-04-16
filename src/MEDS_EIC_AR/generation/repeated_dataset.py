"""Dataset wrapper + collate helper that expands one item into ``n_samples`` interleaved rows.

This is the data-layer side of issue #89: rather than running ``trainer.predict`` once per
``sample`` index and re-prefilling every subject's input ``N`` times, we expand the base dataset to
``len(base) * N`` items where each base item contributes ``N`` *consecutive* rows. Same-subject
rows then end up in adjacent batch positions, which gives us:

1. Tighter padding (rows in the same batch are more likely to share length).
2. Prefix-cache reuse on backends that have one (vLLM/SGLang, see #88 / #97).
3. One ``trainer.predict`` pass instead of ``N`` (saves dataloader/worker spawn + Lightning init).

The wrapper is a thin ``Dataset`` that multiplies the index space and carries per-row metadata
(the ``(subject_idx, sample_idx)`` pair). The collate helper returns a **three-tuple**
``(batch, subject_idxs, sample_idxs)`` rather than attaching metadata as a sidecar attribute on
the batch itself — this keeps the base ``MEDSTorchBatch`` untouched and avoids any coupling to
``meds_torchdata``'s dataclass-mutation behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from meds_torchdata import MEDSTorchBatch


class RepeatedPredictionDataset(Dataset):
    """Wraps a base dataset so each underlying item contributes ``n_samples`` consecutive rows.

    The ordering is **(subject_idx changes slow, sample_idx changes fast)**, so a batch of
    ``batch_size`` covers at most ``ceil(batch_size / n_samples)`` distinct subjects and at least
    ``floor(batch_size / n_samples)`` full sample-groups.

    Concretely, if ``n_samples=4`` and the base dataset is ``[A, B, C, ...]``, this dataset yields::

        A#0 A#1 A#2 A#3 B#0 B#1 B#2 B#3 C#0 ...

    with each ``A#k`` carrying metadata ``(subject_idx=0, sample_idx=k)``. Each call to
    ``__getitem__`` returns a tuple ``(item, subject_idx, sample_idx)`` so the collate function
    downstream can repack the metadata alongside the base batch.

    Examples:
        >>> class FakeDataset:
        ...     def __init__(self, n): self.n = n
        ...     def __len__(self): return self.n
        ...     def __getitem__(self, i): return f"item-{i}"
        >>> base = FakeDataset(3)
        >>> wrapped = RepeatedPredictionDataset(base, n_samples=2)
        >>> len(wrapped)
        6
        >>> for i in range(len(wrapped)):
        ...     print(wrapped[i])
        ('item-0', 0, 0)
        ('item-0', 0, 1)
        ('item-1', 1, 0)
        ('item-1', 1, 1)
        ('item-2', 2, 0)
        ('item-2', 2, 1)

        Validation:

        >>> RepeatedPredictionDataset(base, n_samples=0)
        Traceback (most recent call last):
            ...
        ValueError: n_samples must be a positive integer; got 0.
        >>> RepeatedPredictionDataset(base, n_samples=1.5)
        Traceback (most recent call last):
            ...
        ValueError: n_samples must be a positive integer; got 1.5.
    """

    def __init__(self, base: Dataset, n_samples: int) -> None:
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer; got {n_samples}.")
        self.base = base
        self.n_samples = n_samples

    def __len__(self) -> int:
        return len(self.base) * self.n_samples

    def __getitem__(self, idx: int):
        subject_idx, sample_idx = divmod(idx, self.n_samples)
        return self.base[subject_idx], subject_idx, sample_idx


def collate_with_meta(
    raw_items: Sequence[tuple[object, int, int]],
    base_collate: Callable[[Sequence[object]], MEDSTorchBatch],
) -> tuple[MEDSTorchBatch, torch.Tensor, torch.Tensor]:
    """Wrap a base ``MEDSTorchBatch`` collate to also return per-row metadata.

    ``RepeatedPredictionDataset.__getitem__`` returns ``(item, subject_idx, sample_idx)`` tuples,
    so the dataloader hands ``raw_items`` to this collate as a list of those tuples. We unzip the
    metadata, run the base collate over the items, and return a **three-tuple**
    ``(batch, subject_idxs, sample_idxs)``.

    The three-tuple return (rather than attaching a sidecar attribute to ``batch``) keeps the base
    ``MEDSTorchBatch`` untouched: no ``object.__setattr__`` workaround, no reliance on whether
    ``MEDSTorchBatch`` is a frozen dataclass or accepts extra attributes, and no brittleness if
    ``meds_torchdata`` changes the batch type upstream. The downstream
    :meth:`MEICARModule.predict_step` and the regrouping loop in ``MEICAR_generate_trajectories``
    simply destructure the tuple.

    Args:
        raw_items: Sequence of ``(item, subject_idx, sample_idx)`` tuples from
            :class:`RepeatedPredictionDataset`.
        base_collate: The base dataset's collate function (typically
            ``MEDSPytorchDataset.collate``). Receives just the items, not the metadata.

    Returns:
        A three-tuple ``(batch, subject_idxs, sample_idxs)`` where ``batch`` is whatever the base
        collate produced, ``subject_idxs`` is a ``[B]`` long tensor of the subject indices backing
        each row, and ``sample_idxs`` is a ``[B]`` long tensor of the sample indices.
    """
    items, subject_idxs, sample_idxs = zip(*raw_items, strict=True)
    batch = base_collate(list(items))
    return (
        batch,
        torch.as_tensor(subject_idxs, dtype=torch.long),
        torch.as_tensor(sample_idxs, dtype=torch.long),
    )


__all__ = ["RepeatedPredictionDataset", "collate_with_meta"]

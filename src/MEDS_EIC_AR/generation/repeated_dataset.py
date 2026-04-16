"""Dataset wrapper + collate helper that expands one item into ``n_samples`` interleaved rows.

This is the data-layer side of issue #89: rather than running ``trainer.predict`` once per
``sample`` index and re-prefilling every subject's input ``N`` times, we expand the base dataset to
``len(base) * N`` items where each base item contributes ``N`` *consecutive* rows. Same-subject
rows then end up in adjacent batch positions, which gives us:

1. Tighter padding (rows in the same batch are more likely to share length).
2. Prefix-cache reuse on backends that have one (vLLM/SGLang, see #88 / #97).
3. One ``trainer.predict`` pass instead of ``N`` (saves dataloader/worker spawn + Lightning init).

The wrapper is a thin ``Dataset`` that just multiplies the index space and attaches per-row
metadata (the ``(subject_idx, sample_idx)`` pair) so the downstream regrouping code can unscramble
the flat predictions back into per-sample parquet files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from meds_torchdata import MEDSTorchBatch

# Sentinel attribute name we attach the metadata under on the returned ``MEDSTorchBatch``. Keeping
# this as a module-level constant means the predict_step and reshaping code can refer to a single
# source of truth rather than repeating the string.
META_ATTR = "_repeated_meta"


class RepeatedPredictionDataset(Dataset):
    """Wraps a base dataset so each underlying item contributes ``n_samples`` consecutive rows.

    The ordering is **(subject_idx changes slow, sample_idx changes fast)**, so a batch of
    ``batch_size`` covers at most ``ceil(batch_size / n_samples)`` distinct subjects and at least
    ``floor(batch_size / n_samples)`` full sample-groups.

    Concretely, if ``n_samples=4`` and the base dataset is ``[A, B, C, ...]``, this dataset yields::

        A#0 A#1 A#2 A#3 B#0 B#1 B#2 B#3 C#0 ...

    with each ``A#k`` carrying metadata ``(subject_idx=0, sample_idx=k)``. Each call to
    ``__getitem__`` returns a tuple ``(item, subject_idx, sample_idx)`` so the collate function
    downstream can repack the metadata onto the batch.

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
) -> MEDSTorchBatch:
    """Wrap a base ``MEDSTorchBatch`` collate to also attach per-row metadata.

    ``RepeatedPredictionDataset.__getitem__`` returns ``(item, subject_idx, sample_idx)`` tuples,
    so the dataloader hands ``raw_items`` to this collate as a list of those tuples. We unzip the
    metadata, run the base collate over the items, and stash a ``(subject_idxs, sample_idxs)`` pair
    on the resulting batch under ``META_ATTR`` for the predict step to read.

    ``MEDSTorchBatch`` is a frozen dataclass, so we use ``dataclasses.replace`` to attach the
    metadata via a sidecar dict on a fresh instance — no in-place mutation, no patching of the
    original class.

    Args:
        raw_items: Sequence of ``(item, subject_idx, sample_idx)`` tuples from
            :class:`RepeatedPredictionDataset`.
        base_collate: The base dataset's collate function (typically
            ``MEDSPytorchDataset.collate``). Receives just the items, not the metadata.

    Returns:
        The ``MEDSTorchBatch`` from ``base_collate`` with a sidecar metadata pair stashed under
        ``META_ATTR``. Use :func:`extract_meta` to read it back.
    """
    items, subject_idxs, sample_idxs = zip(*raw_items, strict=True)
    batch = base_collate(list(items))
    meta = (
        torch.as_tensor(subject_idxs, dtype=torch.long),
        torch.as_tensor(sample_idxs, dtype=torch.long),
    )
    # ``MEDSTorchBatch`` is a dataclass without a __dict__ in older versions; use replace + a sidecar
    # set on the returned instance. ``object.__setattr__`` works around frozen=True if applicable.
    object.__setattr__(batch, META_ATTR, meta)
    return batch


def extract_meta(batch: MEDSTorchBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Read back the ``(subject_idxs, sample_idxs)`` metadata that :func:`collate_with_meta` stashed.

    Raises ``AttributeError`` with an actionable message if the metadata is missing — that means
    the batch came through some other collate path and the caller probably has a wiring bug.
    """
    if not hasattr(batch, META_ATTR):
        raise AttributeError(
            f"Batch is missing per-row metadata under {META_ATTR!r}. "
            f"This batch did not come through ``collate_with_meta`` — check the dataloader "
            f"configuration in MEICAR_generate_trajectories."
        )
    return getattr(batch, META_ATTR)


__all__ = ["META_ATTR", "RepeatedPredictionDataset", "collate_with_meta", "extract_meta"]

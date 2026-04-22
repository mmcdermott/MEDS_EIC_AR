"""Dataset wrapper + collate helper that expands one item into ``n_trajectories`` interleaved rows.

This is the data-layer side of issue #89: rather than running ``trainer.predict`` once per
trajectory index and re-prefilling every subject's input ``N`` times, we expand the base dataset
to ``len(base) * N`` items where each base item contributes ``N`` *consecutive* rows. Same-subject
rows then end up in adjacent batch positions, which gives us:

1. Tighter padding (rows in the same batch are more likely to share length).
2. Prefix-cache reuse on backends that have one (vLLM/SGLang, see #88 / #97).
3. One ``trainer.predict`` pass instead of ``N`` (saves dataloader/worker spawn + Lightning init).

The wrapper is a thin ``Dataset`` that multiplies the index space and carries per-row metadata
(the ``(dataset_row_idx, trajectory_idx)`` pair). The collate helper returns a **three-tuple**
``(batch, dataset_row_idxs, trajectory_idxs)`` rather than attaching metadata as a sidecar attribute
on the batch itself — this keeps the base ``MEDSTorchBatch`` untouched and avoids any coupling to
``meds_torchdata``'s dataclass-mutation behavior.

**Terminology.** Throughout this module, ``trajectory_idx`` means "which of the ``n_trajectories``
generated outputs this row corresponds to for its subject" — matching the outer noun in the
Hydra config key ``inference.N_trajectories_per_task_sample`` ("N trajectories per task sample").
The prior iteration of this code used ``sample_idx`` for the same concept; renamed for clarity
since "sample" collides with the generic ML sense of "batch row / datapoint".

``dataset_row_idx`` is the integer index into the *base* (un-expanded) dataset backing each
expanded row — i.e. the value that lets us do ``base[dataset_row_idx]`` to recover the original
item. Importantly, **this is a row index, not a subject identifier**. For a task-labeled
``MEDSPytorchDataset`` the base dataset has one row per ``(subject_id, prediction_time)`` pair,
so a subject with two prediction times occupies two distinct ``dataset_row_idx`` values. The
downstream sort-then-write path relies on ``(dataset_row_idx, trajectory_idx)`` being unique —
that holds by construction here (``divmod`` is bijective), and renaming the field away from the
misleading ``subject_idx`` label documents the invariant in the name itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from meds_torchdata import MEDSTorchBatch

#: Type alias for what the generation dataloader yields (and what ``MEICARModule.predict_step``
#: receives): the base MEDSTorchBatch plus per-row ``(dataset_row_idxs, trajectory_idxs)`` tensors.
PredictBatch = tuple["MEDSTorchBatch", torch.Tensor, torch.Tensor]


class RepeatedPredictionDataset(Dataset):
    """Wraps a base dataset so each underlying item contributes ``n_trajectories`` consecutive rows.

    The ordering is **(dataset_row_idx changes slow, trajectory_idx changes fast)**, so a batch of
    ``batch_size`` covers at most ``ceil(batch_size / n_trajectories)`` distinct subjects and at
    least ``floor(batch_size / n_trajectories)`` full trajectory-groups.

    Concretely, if ``n_trajectories=4`` and the base dataset is ``[A, B, C, ...]``, this dataset
    yields::

        A#0 A#1 A#2 A#3 B#0 B#1 B#2 B#3 C#0 ...

    with each ``A#k`` carrying metadata ``(dataset_row_idx=0, trajectory_idx=k)``. Each call to
    ``__getitem__`` returns a tuple ``(item, dataset_row_idx, trajectory_idx)`` so the collate
    function downstream can repack the metadata alongside the base batch.

    Examples:
        >>> class FakeDataset:
        ...     def __init__(self, n): self.n = n
        ...     def __len__(self): return self.n
        ...     def __getitem__(self, i): return f"item-{i}"
        >>> base = FakeDataset(3)
        >>> wrapped = RepeatedPredictionDataset(base, n_trajectories=2)
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

        >>> RepeatedPredictionDataset(base, n_trajectories=0)
        Traceback (most recent call last):
            ...
        ValueError: n_trajectories must be a positive integer; got 0.
        >>> RepeatedPredictionDataset(base, n_trajectories=1.5)
        Traceback (most recent call last):
            ...
        ValueError: n_trajectories must be a positive integer; got 1.5.
    """

    def __init__(self, base: Dataset, n_trajectories: int) -> None:
        if not isinstance(n_trajectories, int) or n_trajectories < 1:
            raise ValueError(f"n_trajectories must be a positive integer; got {n_trajectories}.")
        self.base = base
        self.n_trajectories = n_trajectories

    def __len__(self) -> int:
        return len(self.base) * self.n_trajectories

    def __getitem__(self, idx: int):
        dataset_row_idx, trajectory_idx = divmod(idx, self.n_trajectories)
        return self.base[dataset_row_idx], dataset_row_idx, trajectory_idx


def collate_with_meta(
    raw_items: Sequence[tuple[object, int, int]],
    base_collate: Callable[[Sequence[object]], MEDSTorchBatch],
) -> PredictBatch:
    """Wrap a base ``MEDSTorchBatch`` collate to also return per-row metadata.

    ``RepeatedPredictionDataset.__getitem__`` returns ``(item, dataset_row_idx, trajectory_idx)``
    tuples, so the dataloader hands ``raw_items`` to this collate as a list of those tuples. We
    unzip the metadata, run the base collate over the items, and return a **three-tuple**
    ``(batch, dataset_row_idxs, trajectory_idxs)``.

    The three-tuple return (rather than attaching a sidecar attribute to ``batch``) keeps the base
    ``MEDSTorchBatch`` untouched: no ``object.__setattr__`` workaround, no reliance on whether
    ``MEDSTorchBatch`` is a frozen dataclass or accepts extra attributes, and no brittleness if
    ``meds_torchdata`` changes the batch type upstream. The downstream
    :meth:`MEICARModule.predict_step` and the regrouping loop in ``MEICAR_generate_trajectories``
    simply destructure the tuple.

    Args:
        raw_items: Sequence of ``(item, dataset_row_idx, trajectory_idx)`` tuples from
            :class:`RepeatedPredictionDataset`.
        base_collate: The base dataset's collate function (typically
            ``MEDSPytorchDataset.collate``). Receives just the items, not the metadata.

    Returns:
        A :data:`PredictBatch` — ``(batch, dataset_row_idxs, trajectory_idxs)`` — where ``batch`` is
        whatever the base collate produced, ``dataset_row_idxs`` is a ``[B]`` long tensor of base-
        dataset row indices backing each row (``base[dataset_row_idxs[i]]`` recovers the original
        item), and ``trajectory_idxs`` is a ``[B]`` long tensor identifying which of the
        ``n_trajectories`` outputs per base-dataset row each batch row corresponds to.
    """
    items, dataset_row_idxs, trajectory_idxs = zip(*raw_items, strict=True)
    batch = base_collate(list(items))
    return (
        batch,
        torch.as_tensor(dataset_row_idxs, dtype=torch.long),
        torch.as_tensor(trajectory_idxs, dtype=torch.long),
    )


__all__ = ["PredictBatch", "RepeatedPredictionDataset", "collate_with_meta"]

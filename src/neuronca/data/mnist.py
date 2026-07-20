"""MNIST datasets and loaders with probe/train split and TaskContext transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from neuronca.data.tasks import TaskContext

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DEFAULT_DATA_ROOT = Path("data/mnist")
DEFAULT_PROBE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 28 * 28


def default_image_transform() -> transforms.Compose:
    """Normalize MNIST with the standard mean/std."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


class TaskAwareDataset(Dataset):
    """Wrap a base dataset and apply TaskContext transforms on the fly.

    Both train and probe subsets share the same ``TaskContext`` instance so an
    atomic ``set_transform`` updates labels/inputs for all consumers.
    """

    def __init__(
        self,
        base: Dataset,
        task_context: TaskContext,
        *,
        return_index: bool = False,
    ) -> None:
        self.base = base
        self.task_context = task_context
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        if len(item) == 2:
            x, y = item
            raw_index = index
        else:
            # Subset of a dataset that already returns (x, y) only.
            x, y = item[0], item[1]
            raw_index = index

        if not torch.is_tensor(y):
            y = torch.tensor(y)
        y = y.long() if y.ndim == 0 or y.dtype != torch.long else y

        x_t = self.task_context.map_input(x if torch.is_tensor(x) else torch.as_tensor(x))
        y_t = self.task_context.map_label(y)
        if not torch.is_tensor(y_t):
            y_t = torch.as_tensor(y_t)
        y_t = y_t.long()

        if self.return_index:
            return x_t, y_t, raw_index
        return x_t, y_t


def _mnist_targets(dataset: datasets.MNIST) -> Tensor:
    # torchvision stores labels as a tensor (or list on very old versions)
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    return targets.long()


def split_train_probe_indices(
    n_train: int,
    probe_size: int = DEFAULT_PROBE_SIZE,
    seed: int = 0,
    *,
    targets: Tensor | None = None,
    num_classes: int = 10,
) -> tuple[Tensor, Tensor]:
    """Return ``(train_indices, probe_indices)`` as long tensors.

    When ``targets`` is provided, the probe set is stratified across classes.
    Otherwise a single seeded random permutation is used.
    """
    if probe_size < 0:
        raise ValueError("probe_size must be non-negative")
    if probe_size > n_train:
        raise ValueError(f"probe_size={probe_size} exceeds n_train={n_train}")

    g = torch.Generator()
    g.manual_seed(int(seed))

    if targets is None or probe_size == 0:
        perm = torch.randperm(n_train, generator=g)
        probe_idx = perm[:probe_size]
        train_idx = perm[probe_size:]
        return train_idx.long(), probe_idx.long()

    targets = targets.long()
    if len(targets) != n_train:
        raise ValueError("targets length must equal n_train")

    # Stratified allocation: floor(probe_size / C) per class, remainder to lowest labels.
    base = probe_size // num_classes
    rem = probe_size % num_classes
    probe_parts: list[Tensor] = []
    train_parts: list[Tensor] = []

    for c in range(num_classes):
        class_idx = torch.nonzero(targets == c, as_tuple=False).view(-1)
        # Shuffle within class with a derived seed for reproducibility
        class_g = torch.Generator()
        class_g.manual_seed(int(seed) * 1_000_003 + c)
        order = torch.randperm(class_idx.numel(), generator=class_g)
        class_idx = class_idx[order]
        take = base + (1 if c < rem else 0)
        if take > class_idx.numel():
            raise ValueError(
                f"class {c} has only {class_idx.numel()} samples; cannot take {take} for probe"
            )
        probe_parts.append(class_idx[:take])
        train_parts.append(class_idx[take:])

    probe_idx = torch.cat(probe_parts) if probe_parts else torch.empty(0, dtype=torch.long)
    train_idx = torch.cat(train_parts) if train_parts else torch.empty(0, dtype=torch.long)

    # Final shuffle so loaders do not see class-contiguous blocks
    probe_idx = probe_idx[torch.randperm(probe_idx.numel(), generator=g)]
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    return train_idx.long(), probe_idx.long()


def load_mnist_raw(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    download: bool = True,
    image_transform: Callable | None = None,
) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Download/load MNIST train and test sets under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tfm = image_transform if image_transform is not None else default_image_transform()
    train = datasets.MNIST(root=str(root), train=True, download=download, transform=tfm)
    test = datasets.MNIST(root=str(root), train=False, download=download, transform=tfm)
    return train, test


def build_mnist_datasets(
    task_context: TaskContext,
    *,
    root: str | Path = DEFAULT_DATA_ROOT,
    probe_size: int = DEFAULT_PROBE_SIZE,
    seed: int = 0,
    download: bool = True,
    return_index: bool = False,
) -> dict[str, Dataset]:
    """Build train / probe / test datasets sharing ``task_context``.

    Probe indices are carved from the official MNIST training split (seeded,
    stratified). The remainder is used for optimization. The official test
    split is reserved for accuracy evaluation only.
    """
    train_full, test_full = load_mnist_raw(root=root, download=download)
    n_train = len(train_full)
    targets = _mnist_targets(train_full)
    train_idx, probe_idx = split_train_probe_indices(
        n_train,
        probe_size=probe_size,
        seed=seed,
        targets=targets,
        num_classes=10,
    )

    train_base = Subset(train_full, train_idx.tolist())
    probe_base = Subset(train_full, probe_idx.tolist())

    train_ds = TaskAwareDataset(train_base, task_context, return_index=return_index)
    probe_ds = TaskAwareDataset(probe_base, task_context, return_index=return_index)
    test_ds = TaskAwareDataset(test_full, task_context, return_index=return_index)

    return {
        "train": train_ds,
        "probe": probe_ds,
        "test": test_ds,
        "train_indices": train_idx,
        "probe_indices": probe_idx,
        "train_full": train_full,
    }


def build_mnist_loaders(
    task_context: TaskContext,
    *,
    root: str | Path = DEFAULT_DATA_ROOT,
    probe_size: int = DEFAULT_PROBE_SIZE,
    seed: int = 0,
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool | None = None,
) -> dict:
    """Build DataLoaders for train / probe / test sharing ``task_context``.

    Returns a dict with loaders plus the index tensors used for the split so
    tests can verify label transforms against raw labels.
    """
    bundles = build_mnist_datasets(
        task_context,
        root=root,
        probe_size=probe_size,
        seed=seed,
        download=download,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def _loader(ds: Dataset, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return {
        "train_loader": _loader(bundles["train"], shuffle=True),
        "probe_loader": _loader(bundles["probe"], shuffle=False),
        "test_loader": _loader(bundles["test"], shuffle=False),
        "train_dataset": bundles["train"],
        "probe_dataset": bundles["probe"],
        "test_dataset": bundles["test"],
        "train_indices": bundles["train_indices"],
        "probe_indices": bundles["probe_indices"],
        "train_full": bundles["train_full"],
        "task_context": task_context,
    }

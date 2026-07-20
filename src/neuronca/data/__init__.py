"""Datasets, loaders, and task transforms."""

from neuronca.data.mnist import (
    MNISTDatasetBundle,
    MNISTLoaderBundle,
    build_mnist_datasets,
    build_mnist_loaders,
    default_image_transform,
    split_train_probe_indices,
)
from neuronca.data.tasks import (
    DigitSplitTransform,
    IdentityTransform,
    LabelPermutationTransform,
    PixelPermutationTransform,
    TaskContext,
    TaskTransform,
    indices_in_digit_range,
    make_digit_split_transform,
    make_identity_transform,
    make_label_permutation_transform,
    make_pixel_permutation_transform,
    make_task_transform,
)

__all__ = [
    "DigitSplitTransform",
    "IdentityTransform",
    "LabelPermutationTransform",
    "MNISTDatasetBundle",
    "MNISTLoaderBundle",
    "PixelPermutationTransform",
    "TaskContext",
    "TaskTransform",
    "build_mnist_datasets",
    "build_mnist_loaders",
    "default_image_transform",
    "indices_in_digit_range",
    "make_digit_split_transform",
    "make_identity_transform",
    "make_label_permutation_transform",
    "make_pixel_permutation_transform",
    "make_task_transform",
    "split_train_probe_indices",
]

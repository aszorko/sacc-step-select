# Adapted from Chen et al. https://github.com/facebookresearch/neural_stpp
# Copyright (c) Facebook, Inc. and its affiliates.

import re
import numpy as np
import torch


class SpatioTemporalDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, test_set, train):
        self.S_mean, self.S_std = self._standardize(train_set)

        S_mean_ = torch.cat([torch.zeros(1, 1).to(self.S_mean), self.S_mean], dim=1)
        S_std_ = torch.cat([torch.ones(1, 1).to(self.S_std), self.S_std], dim=1)
        self.dataset = [(torch.tensor(seq) - S_mean_) / S_std_ for seq in (train_set if train else test_set)]

    def __len__(self):
        return len(self.dataset)

    def _standardize(self, dataset):
        dataset = [torch.tensor(seq) for seq in dataset]
        full = torch.cat(dataset, dim=0)
        S = full[:, 1:]
        S_mean = S.mean(0, keepdims=True)
        S_std = S.std(0, keepdims=True)
        return S_mean, S_std

    def unstandardize(self, spatial_locations):
        return spatial_locations * self.S_std + self.S_mean

    def ordered_indices(self):
        lengths = np.array([seq.shape[0] for seq in self.dataset])
        indices = np.argsort(lengths)
        return indices, lengths[indices]


    def __getitem__(self, index):
        return self.dataset[index]






class EyeTracking2024(SpatioTemporalDataset):

    def __init__(self, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        dataset = np.load("data/waldo/stpp_auto_2024_rescaled.npz")
        exclude_from_train = (dataset.files[0::8] + dataset.files[1::8])
        val_files = dataset.files[0::8]
        test_files = dataset.files[1::8]
        train_set = set(dataset.files).difference(exclude_from_train)
        train_files = [f for f in dataset.files if f in train_set]
        file_splits = {"train": train_files, "val": val_files, "test": test_files}
        train_set = [dataset[f] for f in train_files]
        split_set = [dataset[f] for f in file_splits[split]]
        self.IDs = [f for f in file_splits[split]]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class EyeTrackingRandpix(SpatioTemporalDataset):

    def __init__(self, split="train"):
        assert split in ["train", "val", "test", "all"]
        self.split = split
        dataset = np.load("data/randpix/stpp_auto_randpix_rescaled.npz")
        exclude_from_train = dataset.files[0::4]
        val_files = exclude_from_train[0::2]
        test_files = exclude_from_train[1::2]
        train_set = set(dataset.files).difference(exclude_from_train)
        train_files = [f for f in dataset.files if f in train_set]
        file_splits = {"train": train_files, "val": val_files, "test": test_files, "all": dataset.files}
        train_set = [dataset[f] for f in train_files]
        split_set = [dataset[f] for f in file_splits[split]]
        self.IDs = [f for f in file_splits[split]]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"



def spatiotemporal_events_collate_fn(data):
    """Input is a list of tensors with shape (T, 1 + D)
        where T may be different for each tensor.

    Returns:
        event_times: (N, max_T)
        spatial_locations: (N, max_T, D)
        mask: (N, max_T)
    """
    if len(data) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0) if s.shape[0] != max_len else s for s in data]
    data = torch.stack(padded_seqs, dim=0)
    event_times = data[:, :, 0]
    spatial_locations = data[:, :, 1:]
    mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])
    return event_times, spatial_locations, mask

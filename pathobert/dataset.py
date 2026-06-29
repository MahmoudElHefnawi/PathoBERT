#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Dataset classes for PathoBERT.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class IterableChunkDataset(IterableDataset):
    """
    Iterable dataset for pre-tokenized DNA sequences.

    Features
    --------
    - Memory-mapped NumPy arrays (low RAM usage)
    - DDP-safe sharding
    - Multi-worker DataLoader support
    - Deterministic shuffling
    """

    def __init__(
        self,
        input_ids_path,
        attention_mask_path,
        shuffle=False,
        rank=0,
        world_size=1,
    ):
        self.input_ids_path = input_ids_path
        self.attention_mask_path = attention_mask_path

        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.epoch = 0

    def set_epoch(self, epoch):
        """
        Set current epoch for deterministic shuffling.
        """
        self.epoch = epoch

    def __iter__(self):
        # --------------------------------------------------
        # Memory-mapped loading
        # --------------------------------------------------
        input_ids = np.load(self.input_ids_path, mmap_mode="r")
        attention_mask = np.load(
            self.attention_mask_path,
            mmap_mode="r",
        )

        dataset_len = input_ids.shape[0]

        # --------------------------------------------------
        # DDP-safe partitioning
        # --------------------------------------------------
        dataset_len = (
            dataset_len // self.world_size
        ) * self.world_size

        per_rank = dataset_len // self.world_size

        rank_start = self.rank * per_rank
        rank_end = rank_start + per_rank

        # --------------------------------------------------
        # Multi-worker partitioning
        # --------------------------------------------------
        worker_info = get_worker_info()

        if worker_info is not None:

            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            per_worker = (rank_end - rank_start) // num_workers

            start = rank_start + worker_id * per_worker
            end = start + per_worker

        else:
            start = rank_start
            end = rank_end

        indices = np.arange(start, end)

        # --------------------------------------------------
        # Deterministic shuffle
        # --------------------------------------------------
        if self.shuffle:
            rng = np.random.default_rng(
                self.epoch + self.rank * 1000
            )
            rng.shuffle(indices)

        # --------------------------------------------------
        # Yield tensors
        # --------------------------------------------------
        for i in indices:
            yield {
                "input_ids": torch.from_numpy(
                    input_ids[i].copy()
                ).long(),
                "attention_mask": torch.from_numpy(
                    attention_mask[i].copy()
                ).long(),
            }


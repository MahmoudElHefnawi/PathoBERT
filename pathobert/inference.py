#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Inference utilities for PathoBERT.
"""

from math import ceil

import torch
from tqdm import tqdm


@torch.inference_mode()
def predict(
    model,
    loader,
    total_reads,
    batch_size,
    device,
):
    """
    Perform inference on a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded PathoBERT model.

    loader : DataLoader
        DataLoader returning tokenized batches.

    total_reads : int
        Total number of sequences.

    batch_size : int
        Batch size.

    device : torch.device

    Returns
    -------
    numpy.ndarray
        Prediction probabilities.
    """

    model.eval()

    scores = []

    total_batches = ceil(total_reads / batch_size)

    for batch in tqdm(
        loader,
        total=total_batches,
        desc="Running inference",
        unit="batch",
        dynamic_ncols=True,
    ):

        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        logits = model(
            input_ids,
            attention_mask,
        )

        probs = torch.sigmoid(logits).squeeze(-1)

        scores.append(probs.cpu())

    return torch.cat(scores).numpy()


__all__ = [
    "predict",
]


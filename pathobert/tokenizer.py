#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Tokenizer utilities for PathoBERT.
"""

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
from transformers import AutoTokenizer

from .config import (
    MODEL_NAME,
    USE_FAST_TOKENIZER,
    KMER_SIZE,
    STRIDE,
    MAX_LENGTH,
    NUM_TOKENIZER_WORKERS,
)

# ----------------------------------------------------
# Global tokenizer (loaded once)
# ----------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=USE_FAST_TOKENIZER,
    local_files_only=True,
)

# ----------------------------------------------------
# Tokenize one DNA sequence
# ----------------------------------------------------

def tokenize_seq(
    seq,
    k=KMER_SIZE,
    stride=STRIDE,
    max_length=MAX_LENGTH,
):
    """
    Tokenize one DNA sequence.

    Returns
    -------
    input_ids : np.ndarray
    attention_mask : np.ndarray
    """

    if isinstance(seq, bytes):
        seq = seq.decode()

    kmer_seq = " ".join(
        seq[i:i + k]
        for i in range(
            0,
            len(seq) - k + 1,
            stride,
        )
    )

    encoded = tokenizer(
        kmer_seq,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return (
        encoded["input_ids"].squeeze(0).numpy(),
        encoded["attention_mask"].squeeze(0).numpy(),
    )


# ----------------------------------------------------
# Tokenize a list of sequences
# ----------------------------------------------------

def tokenize_sequences(
    sequences,
    num_workers=NUM_TOKENIZER_WORKERS,
):
    """
    Parallel tokenization.

    Parameters
    ----------
    sequences : list[str]

    Returns
    -------
    input_ids : np.ndarray
    attention_mask : np.ndarray
    """

    func = partial(
        tokenize_seq,
        k=KMER_SIZE,
        stride=STRIDE,
        max_length=MAX_LENGTH,
    )

    with Pool(num_workers) as pool:

        results = list(
            tqdm(
                pool.imap(func, sequences),
                total=len(sequences),
                desc="Tokenizing validation",
            )
        )

    input_ids, attention_mask = zip(*results)

    input_ids = np.stack(input_ids)

    attention_mask = np.stack(attention_mask)

    return input_ids, attention_mask


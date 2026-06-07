#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
from transformers import BertTokenizer
import torch
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# -------------------------------
# Config (MUST match training)
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")

k = 6
stride = 6
max_length = 27

save_dir = "./tokenized_val_npy"
os.makedirs(save_dir, exist_ok=True)

num_workers = min(8, cpu_count())

# -------------------------------
# Tokenize one sequence
# -------------------------------
def tokenize_seq(seq, k=6, stride=6, max_length=27):
    raw_seq = seq.decode() if isinstance(seq, bytes) else seq

    kmer_seq = " ".join(
        raw_seq[j:j+k]
        for j in range(0, len(raw_seq) - k + 1, stride)
    )

    encoded = tokenizer(
        kmer_seq,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return (
        encoded["input_ids"].squeeze(0).numpy(),
        encoded["attention_mask"].squeeze(0).numpy()
    )

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    # Load validation data
    X = np.load("X_valid.npy", mmap_mode="r")
    y = np.load("y_valid.npy", mmap_mode="r")

    print(f"🔹 Validation samples: {len(X)}")

    with Pool(num_workers) as pool:
        func = partial(tokenize_seq, k=k, stride=stride, max_length=max_length)
        results = list(
            tqdm(pool.imap(func, X), total=len(X), desc="Tokenizing validation")
        )

    input_ids_list, attention_mask_list = zip(*results)

    input_ids_array = np.stack(input_ids_list)
    attention_mask_array = np.stack(attention_mask_list)

    # Save
    np.save(os.path.join(save_dir, "input_ids_val.npy"), input_ids_array)
    np.save(os.path.join(save_dir, "attention_mask_val.npy"), attention_mask_array)
    np.save(os.path.join(save_dir, "labels_val.npy"), y)

    print(
        f"💾 Saved validation tensors: "
        f"{input_ids_array.shape}"
    )


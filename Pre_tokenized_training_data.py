#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from transformers import BertTokenizer
import torch
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")

num_chunks = 6
k = 6
stride = 6
max_length = 27  # ~25 k-mers per read
save_dir = "./tokenized_chunks_npy"
os.makedirs(save_dir, exist_ok=True)

num_workers = min(8, cpu_count())  # adjust based on your machine

# -------------------------------
# Tokenize a single sequence
# -------------------------------
def tokenize_seq(seq, k=6, stride=6, max_length=27):
    raw_seq = seq.decode() if isinstance(seq, bytes) else seq
    kmer_seq = " ".join([raw_seq[j:j+k] for j in range(0, len(raw_seq)-k+1, stride)])
    encoded = tokenizer(
        kmer_seq,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encoded["input_ids"].squeeze(0).numpy(), encoded["attention_mask"].squeeze(0).numpy()

# -------------------------------
# Tokenize a full chunk with multi-processing
# -------------------------------
def tokenize_chunk(chunk_index):
    print(f"\n🔹 Processing chunk {chunk_index+1}/{num_chunks}")

    # Load sequences and labels
    X = np.load(f"X_chunk_{chunk_index}.npy", mmap_mode='r')
    y = np.load(f"y_chunk_{chunk_index}.npy", mmap_mode='r')
    total_reads = len(X)

    # Multi-processing pool
    with Pool(num_workers) as pool:
        func = partial(tokenize_seq, k=k, stride=stride, max_length=max_length)
        results = list(tqdm(pool.imap(func, X), total=total_reads, desc="Tokenizing"))

    # Split results
    input_ids_list, attention_mask_list = zip(*results)
    input_ids_array = np.stack(input_ids_list)
    attention_mask_array = np.stack(attention_mask_list)

    # Save as .npy
    np.save(os.path.join(save_dir, f"input_ids_chunk_{chunk_index}.npy"), input_ids_array)
    np.save(os.path.join(save_dir, f"attention_mask_chunk_{chunk_index}.npy"), attention_mask_array)
    np.save(os.path.join(save_dir, f"labels_chunk_{chunk_index}.npy"), y)
    
    print(f"💾 Chunk {chunk_index+1} saved: {input_ids_array.shape[0]} reads × {input_ids_array.shape[1]} tokens")

# -------------------------------
# Main loop
# -------------------------------
if __name__ == "__main__":
    for i in range(num_chunks):
        tokenize_chunk(i)


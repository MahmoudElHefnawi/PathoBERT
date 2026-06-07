#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Bio import SeqIO
import pickle
import numpy as np
import torch



pkl_files = [
    "nonpath_train.pkl",
    "path_train.pkl",
    "nonpath_valid.pkl",
    "path_valid.pkl",
    "nonpath_train_rev.pkl",
    "path_train_rev.pkl",
    "nonpath_valid_rev.pkl",
    "path_valid_rev.pkl",
    "nonpath_sim.pkl",
    "path_sim.pkl",
    "nonpath_valid_sim.pkl",
    "path_valid_sim.pkl",


]

loaded_data = {}

for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        loaded_data[pkl_file] = pickle.load(f)
        print(f"Loaded {pkl_file}: type={type(loaded_data[pkl_file])}, length={len(loaded_data[pkl_file]) if hasattr(loaded_data[pkl_file], '__len__') else 'N/A'}")


# In[ ]:


path_train = loaded_data["path_train.pkl"]
nonpath_train = loaded_data["nonpath_train.pkl"]
path_valid = loaded_data["path_valid.pkl"]
nonpath_valid = loaded_data["nonpath_valid.pkl"]
path_train_rev = loaded_data["path_train_rev.pkl"]
nonpath_train_rev = loaded_data["nonpath_train_rev.pkl"]
path_valid_rev= loaded_data["path_valid_rev.pkl"]
nonpath_valid_rev= loaded_data["nonpath_valid_rev.pkl"]
path_sim= loaded_data["path_sim.pkl"]
nonpath_sim= loaded_data["nonpath_sim.pkl"]
path_valid_sim= loaded_data["path_valid_sim.pkl"]
nonpath_valid_sim= loaded_data["nonpath_valid_sim.pkl"]


path_train_all = path_train + path_train_rev + path_sim
nonpath_train_all = nonpath_train + nonpath_train_rev + nonpath_sim

path_valid_all = path_valid + path_valid_rev + path_valid_sim
nonpath_valid_all = nonpath_valid + nonpath_valid_rev + nonpath_valid_sim

# In[ ]:




# In[ ]:


import random

def stratified_chunks_from_reads(pos_reads, neg_reads, num_chunks):
    # Filter reads to include only those of length 150
    pos_reads = [read for read in pos_reads]
    neg_reads = [read for read in neg_reads]
    
    #random.shuffle(pos_reads)
    #random.shuffle(neg_reads)

    pos_chunks = [pos_reads[i::num_chunks] for i in range(num_chunks)]
    neg_chunks = [neg_reads[i::num_chunks] for i in range(num_chunks)]

    balanced_chunks = []
    for i in range(num_chunks):
        reads = pos_chunks[i] + neg_chunks[i]
        labels = [1.0] * len(pos_chunks[i]) + [0.0] * len(neg_chunks[i])
        combined = list(zip(reads, labels))
        random.shuffle(combined)

        reads_shuffled, labels_shuffled = zip(*combined)
        balanced_chunks.append((list(reads_shuffled), list(labels_shuffled)))

    return balanced_chunks


# In[ ]:


import numpy as np

# Suppose pos_reads and neg_reads are loaded lists of sequences
train_chunks = stratified_chunks_from_reads(path_train_all, nonpath_train_all, num_chunks=6)

# Save each chunk
for i, (X_chunk, y_chunk) in enumerate(train_chunks):
    np.save(f"X_chunk_{i}.npy", X_chunk)
    np.save(f"y_chunk_{i}.npy", np.array(y_chunk))


# In[ ]:


total_samples = sum(len(X) for X, _ in train_chunks)
print(f'total_samples: {total_samples}')


# In[ ]:


import numpy as np

# Generate 1 chunk of validation data
valid_chunks = stratified_chunks_from_reads(path_valid_all, nonpath_valid_all, num_chunks=1)

# There will be only one chunk, so unpack it directly
X_valid, y_valid = valid_chunks[0]

# Save as .npy
np.save("X_valid.npy", X_valid)               # Strings or sequences
np.save("y_valid.npy", np.array(y_valid))     # Labels as NumPy array

# Print total samples
total_samples_valid = len(X_valid)
print(f"total_samples_valid: {total_samples_valid}")


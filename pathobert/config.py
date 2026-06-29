#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
PathoBERT configuration.

Global constants used throughout the package.
"""

import os
from multiprocessing import cpu_count

# ==========================================================
# Package
# ==========================================================

__version__ = "1.0.0"

PACKAGE_NAME = "pathobert"

# ==========================================================
# Directories
# ==========================================================

OUTPUT_DIR = "./output"
TOKENIZED_DIR = "./tokenized"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TOKENIZED_DIR, exist_ok=True)

# ==========================================================
# DNABERT
# ==========================================================

MODEL_NAME = "zhihan1996/DNA_bert_6"

# If True, load only from the local Hugging Face cache.
LOCAL_FILES_ONLY = False

# torch.compile() during inference
USE_TORCH_COMPILE = False

# ==========================================================
# Model
# ==========================================================

MODEL_DTYPE = "float32"

DEFAULT_THRESHOLD = 0.5

# ==========================================================
# Tokenization
# ==========================================================

KMER_SIZE = 6
STRIDE = 6
MAX_LENGTH = 27

USE_FAST_TOKENIZER = True

# ==========================================================
# Pretokenized tensor files
# ==========================================================

MMAP_MODE = "r"

INPUT_IDS_FILE = os.path.join(
    TOKENIZED_DIR,
    "input_ids.npy",
)

ATTENTION_MASK_FILE = os.path.join(
    TOKENIZED_DIR,
    "attention_mask.npy",
)

# ==========================================================
# DataLoader
# ==========================================================

DEFAULT_BATCH_SIZE = 32

DEFAULT_DATALOADER_WORKERS = min(8, cpu_count())

# ==========================================================
# Tokenization workers
# ==========================================================

NUM_TOKENIZER_WORKERS = min(8, cpu_count())

# ==========================================================
# CPU
# ==========================================================

CPU_THREADS = cpu_count()

INTEROP_THREADS = min(4, CPU_THREADS)

# ==========================================================
# Output files
# ==========================================================

DEFAULT_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions.tsv",
)

DEFAULT_PROBS_FILE = os.path.join(
    OUTPUT_DIR,
    "probs.pt",
)

DEFAULT_STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "stats.pt",
)

DEFAULT_PLOT_FILE = os.path.join(
    OUTPUT_DIR,
    "Read-level_prediction_distribution.png",
)


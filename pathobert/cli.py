#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3

"""
Command-line interface for PathoBERT.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATALOADER_WORKERS,
    INPUT_IDS_FILE,
    ATTENTION_MASK_FILE,
)

from .fasta import read_fasta
from .tokenizer import tokenize_sequences
from .dataset import IterableChunkDataset
from .model import load_model
from .inference import predict
from .analysis import analyze_predictions


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pathobert",
        description="PathoBERT read-level bacterial pathogenicity prediction",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained checkpoint (.pt)",
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_DATALOADER_WORKERS,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--out",
        default="predictions.tsv",
        help="Output TSV file",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not display histogram",
    )

    return parser


def main():

    args = build_parser().parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("📂 Reading FASTA...")
    ids, sequences = read_fasta(args.fasta)

    print("🧬 Tokenizing...")
    tokenize_sequences(sequences)

    dataset = IterableChunkDataset(
        input_ids_path=INPUT_IDS_FILE,
        attention_mask_path=ATTENTION_MASK_FILE,
        shuffle=False,
        rank=0,
        world_size=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("📥 Loading model...")
    model = load_model(args.model, device)

    print("🚀 Running inference...")
    probs = predict(
        model=model,
        loader=loader,
        total_reads=len(ids),
        batch_size=args.batch_size,
        device=device,
    )

    print("📊 Analyzing predictions...")
    analyze_predictions(
        probs,
        threshold=args.threshold,
        show_plot=not args.no_plot,
    )

    print("✅ Finished.")


if __name__ == "__main__":
    main()


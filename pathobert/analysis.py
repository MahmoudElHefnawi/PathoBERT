#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
analysis.py

Prediction analysis utilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import (
    DEFAULT_THRESHOLD,
    DEFAULT_PLOT_FILE,
)


def analyze_predictions(
    probs,
    threshold=DEFAULT_THRESHOLD,
    plot_file=DEFAULT_PLOT_FILE,
    show_plot=True,
):
    """
    Analyze read-level prediction probabilities.

    Parameters
    ----------
    probs : str | torch.Tensor | np.ndarray | list
        Path to .pt file or probability array.

    threshold : float
        Decision threshold.

    plot_file : str
        Output histogram filename.

    show_plot : bool
        Whether to display the figure.

    Returns
    -------
    dict
        Summary statistics.
    """

    # Load probabilities if a filename is supplied
    if isinstance(probs, str):
        probs = torch.load(probs)

    # Convert to NumPy
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    else:
        probs = np.asarray(probs)

    mean_score = np.mean(probs)

    # Plot histogram
    plt.figure(figsize=(7, 4))

    plt.hist(
        probs,
        bins=50,
        alpha=0.8,
        edgecolor="black",
    )

    plt.axvline(
        mean_score,
        color="blue",
        linestyle="--",
        label=f"Mean = {mean_score:.3f}",
    )

    plt.axvline(
        threshold,
        color="black",
        linestyle="-",
        label=f"Threshold ({threshold})",
    )

    plt.title("Read-level prediction distribution")
    plt.xlabel("Predicted pathogenicity score")
    plt.ylabel("Number of reads")

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Statistics
    read_labels = (probs >= threshold).astype(np.int32)

    genome_prediction = int(read_labels.mean() > 0.5)

    path_fraction = read_labels.mean()
    nonpath_fraction = 1.0 - path_fraction

    print("=" * 50)
    print(f"Mean score           : {mean_score:.6f}")
    print(f"Genome prediction    : {genome_prediction}")
    print(f"Pathogenic fraction (PathFrac) : {path_fraction:.3%}")
    print(f"Non-pathogenic votes : {nonpath_fraction:.3%}")
    print("=" * 50)

    return {
        "mean": float(mean_score),
        "genome_prediction": genome_prediction,
        "path_fraction (PathFrac)": float(path_fraction),
        "nonpath_fraction": float(nonpath_fraction),
    }


__all__ = [
    "analyze_predictions",
]


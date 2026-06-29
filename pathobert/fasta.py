#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
FASTA reader for PathoBERT.

No external dependencies.
"""


def read_fasta(path):
    """
    Read a FASTA file.

    Parameters
    ----------
    path : str
        Path to FASTA file.

    Returns
    -------
    ids : list[str]
        Sequence identifiers.

    sequences : list[str]
        DNA sequences.
    """

    ids = []
    sequences = []

    with open(path, "r") as f:
        seq = []
        seq_id = None

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if seq_id is not None:
                    ids.append(seq_id)
                    sequences.append("".join(seq).upper())

                seq_id = line[1:].split()[0]   # first token only
                seq = []

            else:
                seq.append(line)

        if seq_id is not None:
            ids.append(seq_id)
            sequences.append("".join(seq).upper())

    return ids, sequences


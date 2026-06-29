# PathoBERT

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PathoBERT is a deep learning framework for **read-level bacterial pathogenicity prediction** from DNA sequencing reads.

The model combines:

- DNABERT
- LoRA (Low-Rank Adaptation)
- CNN feature extraction
- MCBAM attention
- Multi-Scale Convolutional Attention (MSCA)

to provide robust pathogenicity prediction from short sequencing reads.

---

# Features

- Read FASTA files
- Automatic k-mer tokenization
- DNABERT backbone
- LoRA-based fine-tuning
- CNN + MCBAM + MSCA classifier
- GPU and CPU inference
- Batch inference
- Probability output
- Genome-level prediction summary
- Publication-quality prediction histogram

---

# Installation

## From PyPI

```bash
pip install pathobert --no-deps
pip install -r requirements.txt
```

## From GitHub

```bash
git clone https://github.com/salimalaarag/PathoBERT.git

cd PathoBERT

pip install .
```

or

```bash
pip install git+https://github.com/salimalaarag/PathoBERT.git
```

---

# Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2
- transformers
- peft
- numpy
- pandas
- matplotlib
- tqdm

---

# Download DNABERT

The first run automatically downloads

```
zhihan1996/DNA_bert_6
```

from Hugging Face.

For offline usage:

```python
from transformers import BertModel

BertModel.from_pretrained(
    "zhihan1996/DNA_bert_6"
)
```

or set

```python
LOCAL_FILES_ONLY=True
```

inside

```
pathobert/config.py
```

---

# Usage

## Command Line

```bash
pathobert \
    --fasta reads.fasta \
    --checkpoint final_model.pt
```

Example

```bash
pathobert \
    --fasta ecoli_reads.fasta \
    --checkpoint final_model.pt \
    --batch-size 64
```

---

# Python API

```python
from pathobert.model import load_model
from pathobert.inference import predict
```

---

# Output

PathoBERT generates

```
output/
│
├── predictions.tsv
├── probs.pt
├── stats.pt
└── Read-level_prediction_distribution.png
```

---

# Model Architecture

```
DNA Reads
      │
      ▼
K-mer Tokenization
      │
      ▼
DNABERT
      │
      ▼
LoRA
      │
      ▼
CNN
      │
      ▼
MCBAM
      │
      ▼
MSCA
      │
      ▼
Classification Head
      │
      ▼
Pathogenicity Probability
```

---

# Repository Structure

```
pathobert/

├── __init__.py
├── analysis.py
├── cli.py
├── config.py
├── dataset.py
├── fasta.py
├── inference.py
├── model.py
└── tokenizer.py
```

---

# Citation

If you use PathoBERT in your research, please cite:

```
Salem A. El-aarag.

PathoBERT:
Read-Level Bacterial Pathogenicity Prediction Using
DNABERT, LoRA, CNN, MCBAM and Multi-Scale Convolutional Attention.

(Manuscript in preparation)
```

Update this section after publication.

---

# License

MIT License.

---

# Research Use

PathoBERT is intended **for research purposes only**.

It is **not** intended for:

- clinical diagnosis
- medical decision making
- patient care
- regulatory use

Users are responsible for validating results before applying them in biological or clinical studies.

---

# Author

**Salem A. El-aarag**

Genome Environment and Bioinformatics Research Laboratory (GEBRI)

Email:
- salem.abdelmonem.stu@gebri.usc.edu.eg
- salim.alaarag@yahoo.com

GitHub

https://github.com/salimalaarag

---

# Acknowledgements

This work uses

- DNABERT
- Hugging Face Transformers
- PEFT
- PyTorch

We thank the developers of these open-source projects.

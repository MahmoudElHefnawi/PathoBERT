#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from peft import LoraConfig, get_peft_model, TaskType
from torch.nn import BCEWithLogitsLoss
import math
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch.multiprocessing as mp
#mp.set_sharing_strategy("file_system")


import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import psutil, os
import re
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR
import json
import pickle
import torch.nn as nn
import multiprocessing
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import gc
import torch.multiprocessing as mp
from multiprocessing import Manager
import time
import socket
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import json
import pickle
from peft import LoraConfig, get_peft_model, PeftModel
import os
import torch
import torch.distributed as dist
import socket

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    matthews_corrcoef, f1_score,
    average_precision_score,
    confusion_matrix
)

import torch.nn as nn

import torch
import torch.nn as nn

class IterableChunkDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset for pre-tokenized data with correct DDP sharding.
    Each rank sees exactly the same number of samples.
    """
    def __init__(
        self,
        input_ids_path,
        attention_mask_path,
        labels_path,
        shuffle=False,
        rank=0,
        world_size=1
    ):
        self.input_ids_path = input_ids_path
        self.attention_mask_path = attention_mask_path
        self.labels_path = labels_path
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import numpy as np
        from torch.utils.data import get_worker_info

        # --- 🔹 Memory-mapped loading ---
        input_ids = np.load(self.input_ids_path, mmap_mode="r")
        attention_mask = np.load(self.attention_mask_path, mmap_mode="r")
        labels = np.load(self.labels_path, mmap_mode="r")

        dataset_len = len(labels)

        # --- 🔹 DDP-safe truncation ---
        dataset_len = (dataset_len // self.world_size) * self.world_size
        per_rank = dataset_len // self.world_size

        rank_start = self.rank * per_rank
        rank_end = rank_start + per_rank

        # --- 🔹 Worker-level split (if num_workers > 0) ---
        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = (rank_end - rank_start) // num_workers
            start = rank_start + worker_id * per_worker
            end = start + per_worker
        else:
            start, end = rank_start, rank_end

        indices = np.arange(start, end)

        # --- 🔹 Deterministic shuffling ---
        if self.shuffle:
            rng = np.random.default_rng(self.epoch + self.rank * 1000)
            rng.shuffle(indices)

        # --- 🔹 Yield pre-tokenized tensors ---
        for i in indices:
            yield {
                "input_ids": torch.from_numpy(input_ids[i]).long(),
                "attention_mask": torch.from_numpy(attention_mask[i]).long(),
                "labels": torch.tensor(labels[i], dtype=torch.float32),
            }


# In[ ]:


def worker_init_fn(worker_id):
    # Seed each worker deterministically
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# In[ ]:


class MCBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial = nn.Sequential(nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(1))


    def forward(self, x):
        # Channel attention
        ca = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        x = x * torch.sigmoid(ca)

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg, mx], dim=1)
        x = x * torch.sigmoid(self.spatial(sa))

        return x
        
class MSCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv3 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, 5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, 7, padding=3)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels * 3, out_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels * 3, 1),
            nn.Sigmoid()
        )

        self.project = nn.Conv1d(out_channels * 3, out_channels, 1)

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)

        feats = torch.cat([f3, f5, f7], dim=1)
        attn = self.attention(feats)
        feats = feats * attn

        return self.project(feats)

def load_dnabert_with_lora(
    model_name="zhihan1996/DNA_bert_6",
    r=16,
    alpha=32,
    dropout=0.05
):
    base_model = BertModel.from_pretrained(model_name)

    # Freeze base BERT
    for p in base_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["query", "key", "value"]
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model

class DNABERT_CNN_MCBAM_MSCA(nn.Module):
    def __init__(
        self,
        cnn_channels=256,
        msca_channels=256,
        num_classes=1,
        dropout=0.3
    ):
        super().__init__()

        self.bert = load_dnabert_with_lora()
        hidden_size = self.bert.config.hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, cnn_channels, 7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU()
        )

        self.mcbam = MCBAM(cnn_channels)
        self.msca = MSCA(cnn_channels, msca_channels)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(msca_channels, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        x = x[:, 1:-1, :]
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.mcbam(x)
        x = self.msca(x)
        x = self.pool(x).squeeze(-1)

        return self.head(x)

class DNABERT_CNN_MCBAM_MSCA(nn.Module):
    def __init__(self, cnn_channels=256, msca_channels=256, num_classes=1, dropout=0.3):
        super().__init__()

        self.bert = load_dnabert_with_lora()
        hidden_size = self.bert.config.hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, cnn_channels, 7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU()
        )

        self.mcbam = MCBAM(cnn_channels)
        self.msca = MSCA(cnn_channels, msca_channels)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(msca_channels, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_features=False):
        feats = {}

        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x[:, 1:-1, :]
        feats["bert"] = x.detach()

        x = x.transpose(1, 2)
        x = self.cnn(x)
        feats["cnn"] = x.detach()

        x = self.mcbam(x)
        feats["mcbam"] = x.detach()

        x = self.msca(x)
        feats["msca"] = x.detach()

        pooled = self.pool(x).squeeze(-1)
        feats["pooled"] = pooled.detach()

        logits = self.head(pooled)

        if return_features:
            return logits, feats

        return logits


def fisher_score(feats, labels):
    pos = feats[labels == 1]
    neg = feats[labels == 0]

    mean_diff = torch.norm(pos.mean(0) - neg.mean(0))**2
    var = pos.var(0).mean() + neg.var(0).mean()

    return (mean_diff / (var + 1e-8)).item()
# In[ ]:

start_time = time.time() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DNABERT_CNN_MCBAM_MSCA()
model = model.to(device)

checkpoint = torch.load("final_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
world_size=1
num_workers = min(2, multiprocessing.cpu_count() // world_size)
model.eval()
batch_size=256

# In[ ]:


val_dataset = IterableChunkDataset(
    input_ids_path="./tokenized_val_npy/input_ids_val.npy",
    attention_mask_path="./tokenized_val_npy/attention_mask_val.npy",
    labels_path="./tokenized_val_npy/labels_val.npy",
    shuffle=False,
    rank=0,
    world_size=1   # ✅ important
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn,
    pin_memory=True
)


# In[ ]:


from sklearn.metrics import roc_auc_score

all_feats = {
    "bert": [],
    "cnn": [],
    "mcbam": [],
    "msca": [],
    "pooled": []
}

y_true, y_scores = [], []

model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits, feats = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_features=True
        )

        probs = torch.sigmoid(logits.view(-1))

        y_scores.append(probs.cpu())
        y_true.append(labels.cpu())

        # 🔥 store features
        for k in all_feats:
            f = feats[k]

            if f.dim() == 3:
                f = f.mean(dim=-1)   # [B, C]

            all_feats[k].append(f.cpu())
        del batch, logits, probs

torch.cuda.empty_cache()
gc.collect()


# In[ ]:


y = torch.cat(y_true).numpy()
y_scores = torch.cat(y_scores).numpy()

auroc = roc_auc_score(y, y_scores)
print(f"✅ AUROC: {auroc:.4f}")

for k in all_feats:
    all_feats[k] = torch.cat(all_feats[k])


    
print("\n===== LAYER-WISE SEPARATION =====")

for k in all_feats:
    score = fisher_score(all_feats[k], y)
    print(f"{k}: {score:.4f}")
    


# In[ ]:


import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    matthews_corrcoef, f1_score,
    average_precision_score,
    confusion_matrix
)

auroc = roc_auc_score(y, y_scores)
print(f"✅ AUROC: {auroc:.4f}")

# 6. Threshold tuning (MCC)
# -------------------------
thresholds = np.linspace(0.0, 1.0, 200)

best_mcc = -1
best_t = 0.5

for t in thresholds:
    y_pred = (y_scores > t).astype(int)
    mcc = matthews_corrcoef(y, y_pred)

    if mcc > best_mcc:
        best_mcc = mcc
        best_t = t

print("\n===== THRESHOLD =====")
print(f"Best threshold: {best_t:.4f}")
print(f"Best MCC: {best_mcc:.4f}")

# -------------------------
# 7. Final predictions
# -------------------------
y_pred = (y_scores > best_t).astype(int)

# -------------------------
# 8. Metrics
# -------------------------
acc = accuracy_score(y, y_pred)
auroc = roc_auc_score(y, y_scores)
mcc = matthews_corrcoef(y, y_pred)
f1 = f1_score(y, y_pred)
aupr = average_precision_score(y, y_scores)

print("\n===== FINAL METRICS =====")
print(f"Accuracy: {acc:.4f}")
print(f"AUROC:   {auroc:.4f}")
print(f"MCC:     {mcc:.4f}")
print(f"F1:      {f1:.4f}")
print(f"AUPR:    {aupr:.4f}")

# -------------------------
# 9. Confusion matrix
# -------------------------
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

print("\n===== CONFUSION MATRIX =====")
print(f"TP: {tp}, FP: {fp}")
print(f"FN: {fn}, TN: {tn}")

# -------------------------
# 10. Extra insight (VERY USEFUL)
# -------------------------
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)

print("\n===== EXTRA =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")


# In[ ]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_layer(feats, labels, name):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(feats.numpy())

    plt.figure()
    plt.scatter(proj[labels==0, 0], proj[labels==0, 1], alpha=0.3, label="Neg")
    plt.scatter(proj[labels==1, 0], proj[labels==1, 1], alpha=0.3, label="Pos")
    plt.legend()
    plt.title(name)
    plt.savefig(name)


for k in all_feats:
    plot_layer(all_feats[k], y, k)   


# In[ ]:


def fisher_score_normalized(feats, labels):
    # normalize each sample
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)

    pos = feats[labels == 1]
    neg = feats[labels == 0]

    mean_diff = torch.norm(pos.mean(0) - neg.mean(0))**2
    var = pos.var(0).mean() + neg.var(0).mean()

    return (mean_diff / (var + 1e-8)).item()

print("\n===== LAYER-WISE SEPARATION =====")

for k in all_feats:
    score = fisher_score_normalized(all_feats[k], y)
    print(f"{k}: {score:.4f}")
    


# In[ ]:


def cosine_separation(feats, labels):
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)

    pos_mean = feats[labels == 1].mean(0)
    neg_mean = feats[labels == 0].mean(0)

    return torch.dot(pos_mean, neg_mean).item()
print("\n===== LAYER-WISE SEPARATION =====")
    
for k in all_feats:
    score = cosine_separation(all_feats[k], y)
    print(f"{k}: {score:.4f}")    
    


# In[ ]:


from scipy.stats import wasserstein_distance
def wasserstein_sep(feats, labels):
    pos = feats[labels == 1].mean(dim=1).cpu().numpy()
    neg = feats[labels == 0].mean(dim=1).cpu().numpy()

    return wasserstein_distance(pos, neg)

print("\n===== LAYER-WISE SEPARATION =====")
    
for k in all_feats:
    score = wasserstein_sep(all_feats[k], y)
    print(f"{k}: {score:.4f}")    
        


# In[ ]:


def separation_ratio(feats, labels):
    pos = feats[labels == 1]
    neg = feats[labels == 0]

    inter = torch.norm(pos.mean(0) - neg.mean(0))
    intra = pos.std(0).mean() + neg.std(0).mean()

    return (inter / (intra + 1e-8)).item()

print("\n===== LAYER-WISE SEPARATION =====")
    
for k in all_feats:
    score = separation_ratio(all_feats[k], y)
    print(f"{k}: {score:.4f}")    
            


# In[ ]:


from sklearn.linear_model import LogisticRegression

def linear_probe(feats, labels):

    # convert safely
    if isinstance(feats, torch.Tensor):
        X = feats.detach().cpu().numpy()
    else:
        X = feats

    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy()
    else:
        y = labels

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    return clf.score(X, y)

for k in all_feats:
    score = linear_probe(all_feats[k], y)
    print(f"{k}: {score:.4f}")    
                


# In[ ]:


from sklearn.metrics import silhouette_score
import torch
import numpy as np

def silhouette(feats, labels):

    # convert feats safely
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()

    # convert labels safely
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    return silhouette_score(feats, labels)
for k in all_feats:
    score = silhouette(all_feats[k], y)
    print(f"{k}: {score:.4f}")    
                    


# In[ ]:


def classification_margin(logits, labels):
    probs = torch.sigmoid(logits)

    pos_margin = probs[labels==1].mean()
    neg_margin = probs[labels==0].mean()

    return (pos_margin - neg_margin).item()
for k in all_feats:
    score = classification_margin(all_feats[k], y)
    print(f"{k}: {score:.4f}")    
                        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





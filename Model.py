#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
import torchmetrics
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import torch.distributed as dist
from torchmetrics import MeanMetric, Accuracy
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
from torchmetrics.classification import AUROC
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
base_model = BertForSequenceClassification.from_pretrained("zhihan1996/DNA_bert_6", num_labels=2)
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn import BCEWithLogitsLoss
import math
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


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


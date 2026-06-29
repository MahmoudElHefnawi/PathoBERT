#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

from transformers import BertModel
from peft import (
    LoraConfig,
    get_peft_model,
)

from .config import (
    MODEL_NAME,
    LOCAL_FILES_ONLY,
    USE_TORCH_COMPILE,
    MODEL_DTYPE,
)

# ==========================================================
# MCBAM
# ==========================================================

class MCBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )

        self.spatial = nn.Sequential(
            nn.Conv1d(
                2,
                1,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(1),
        )

    def forward(self, x):
        ca = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        x = x * torch.sigmoid(ca)

        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        sa = torch.cat([avg, mx], dim=1)
        x = x * torch.sigmoid(self.spatial(sa))

        return x


# ==========================================================
# MSCA
# ==========================================================

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
            nn.Sigmoid(),
        )

        self.project = nn.Conv1d(
            out_channels * 3,
            out_channels,
            1,
        )

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)

        feats = torch.cat([f3, f5, f7], dim=1)

        attn = self.attention(feats)
        feats = feats * attn

        return self.project(feats)


# ==========================================================
# DNABERT + LoRA
# ==========================================================

def load_dnabert_with_lora(
    model_name=MODEL_NAME,
    r=16,
    alpha=32,
    dropout=0.05,
):
    base_model = BertModel.from_pretrained(
        model_name,
        local_files_only=LOCAL_FILES_ONLY,
    )

    # Freeze DNABERT
    base_model.requires_grad_(False)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "query",
            "key",
            "value",
        ],
    )

    return get_peft_model(base_model, lora_config)


# ==========================================================
# PathoBERT
# ==========================================================

class DNABERT_CNN_MCBAM_MSCA(nn.Module):
    def __init__(
        self,
        cnn_channels=256,
        msca_channels=256,
        num_classes=1,
        dropout=0.3,
    ):
        super().__init__()

        self.bert = load_dnabert_with_lora()

        hidden_size = self.bert.config.hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                cnn_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        self.mcbam = MCBAM(cnn_channels)
        self.msca = MSCA(cnn_channels, msca_channels)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                msca_channels,
                num_classes,
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        # Remove CLS and SEP
        x = x[:, 1:-1, :]

        # (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        x = self.cnn(x)
        x = self.mcbam(x)
        x = self.msca(x)

        x = self.pool(x).squeeze(-1)

        return self.head(x)


# ==========================================================
# Load trained checkpoint
# ==========================================================

def load_model(
    checkpoint_path,
    device=None,
):
    """
    Load a trained PathoBERT checkpoint.

    Supports both:
        {"model_state": state_dict}
    and:
        state_dict
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = DNABERT_CNN_MCBAM_MSCA().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)

    if MODEL_DTYPE == "float16":
        model = model.half()
    elif MODEL_DTYPE == "bfloat16":
        model = model.bfloat16()

    model.eval()

    if USE_TORCH_COMPILE:
        model = torch.compile(model)

    return model


__all__ = [
    "MCBAM",
    "MSCA",
    "load_dnabert_with_lora",
    "DNABERT_CNN_MCBAM_MSCA",
    "load_model",
]


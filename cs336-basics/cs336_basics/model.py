#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


class TextClassifier(nn.Module):
    """A Transformer-based text classifier.

    Args:
        vocab_size: int
            The number of unique items in the vocabulary.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention.
        d_ff: int
            Dimensionality of the feed-forward inner layer.
        num_classes: int
            Number of classification categories (default: 2 for binary classification).
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities with this rate.
        residual_pdrop: Optional[float], default is None.
            If given, apply dropout to the sum of the token and position embeddings
            and to the output of each sub-layer.

    Returns:
        FloatTensor of shape (batch size, num_classes) with the classification logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        num_classes: int = 2,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        
        # Instead of a language model head, add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )
        
        self.residual_pdrop = residual_pdrop
        # report number of parameters
        logger.info(
            "number of parameters: %.2fM" % (self.get_num_params() / 1e6,)
        )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
        return n_params

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: LongTensor of shape `(batch_size, sequence_length)`.
                Input token IDs.

        Returns: A FloatTensor of shape
            (batch size, num_classes) with the classification logits.
        """
        batch_size, sequence_length = x.size()
        
        # (batch size, sequence_length, d_model)
        embedded_tokens = self.token_embeddings(x)

        # Shape: (1, sequence_length)
        positions = torch.arange(
            0, sequence_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        # (1, sequence_length, d_model)
        embedded_positions = self.position_embeddings(positions)
        # (batch size, sequence_length, d_model)
        x = embedded_tokens + embedded_positions
        if self.residual_pdrop:
            # (batch size, sequence_length, d_model)
            x = F.dropout(x, self.residual_pdrop)
        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)
        
        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        
        # Global average pooling over sequence dimension
        # (batch size, d_model)
        x = x.mean(dim=1)
        
        # (batch size, num_classes)
        logits = self.classifier(x)
        
        return logits

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """A single Transformer layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_pdrop if attn_pdrop else 0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model)
        self.residual_pdrop = residual_pdrop

    def forward(self, x: torch.Tensor):
        # Apply the multi-head self-attention sublayer
        x_ln = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x_attn = self.attn(
            x_ln, x_ln, x_ln, need_weights=False, attn_mask=causal_mask, is_causal=True
        )[0]
        if self.residual_pdrop is not None:
            x_attn = F.dropout(x_attn, self.residual_pdrop)
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        if self.residual_pdrop is not None:
            x_ffn = F.dropout(x_ffn, self.residual_pdrop)
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x
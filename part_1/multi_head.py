import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import causal_mask

class MultiHeadSelfAttention(nn.Module):
    """"1.4 Multi-head attention with explicit shape tracing.

    Dimensions (before masking):
    x: (B,T,d_model)
    qkv: (B,T,3*d_model)
    view→ (B,T,3,head,d_head) where d_head = d_model // heads
    split→ q,kv: (B, n_heads, T, d_head)
    swap→ (B, n_head, T, d_head)
    scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
    weights: (B, n_head, T, T) = softmax(scores)
    ctx: (B, n_head, T, d_head) = weights @ v
    emerge = (B, T, n_head*d_head) = (B, T, d_model)
    """

def __init__(self, d_model, n_head: int, dropout: float =0.0, trace_shape: bool = True):
    super().__init__()
    assert d_model % n_head == 0, "d_model must be divisible by n_head"
    self.n_head = n_head
    self.d_head = d_model // n_head
    self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
    self.trace_shape = trace_shape
    

def forward(self, x: torch.Tensor): # (B,T,d_model)
    B, T, C = x.shape
    qkv = self.qkv(x) # (B,T,3*d_model)
    qkv = qkv.view(B, T, 3, self.n_head, self.d_head) # (B,T,3,heads,d_heads)
    if self.trace_shape:
        print("qkv view", qkv.shape)
    q, k,v = = 
import os
import sys
# import time
from timeit import default_timer
import numpy as np
import argparse
from typing import Callable
import pandas as pd
from collections import defaultdict

import random
import math
import torch
import torch.nn as nn
from einops import rearrange,einsum

# nvtx
import torch.cuda.nvtx as nvtx

# cs336_basics
from cs336_basics.module import scaled_dot_product_attention_func
from cs336_basics.optim import cross_entropy_func, AdamW

# device
from contextlib import nullcontext
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# cudnn
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

def pytorch_flashattention_backward(ctx, grad_out):
    Q, K, V, O, L = ctx.saved_tensors
    B, N_QUERIES, d = Q.shape
    _, N_KEYS, _ = K.shape       

    # S = QKᵀ / √d 
    if ctx.is_causal:
        mask = torch.tril(torch.ones(N_QUERIES, N_KEYS, dtype=torch.float32, device=Q.device))
        S = einsum(Q, K, "B N_QUERIES d, B N_KEYS d -> B N_QUERIES N_KEYS").masked_fill(mask == 0, float("-inf")) / math.sqrt(d)
    else:
        S = einsum(Q, K, "B N_QUERIES d, B N_KEYS d -> B N_QUERIES N_KEYS") / math.sqrt(d)
        
    # P_ij = exp(S_ij - L_i)  N X N_QUERIES X N_KEYS 
    P = torch.exp(S - L.unsqueeze(-1))
    # dV = Pᵀ dO
    dV = einsum(P, grad_out, "B N_QUERIES N_KEYS, B N_QUERIES d -> B N_KEYS d")
    # dP = dO Vᵀ
    dP = einsum(grad_out, V, "B N_QUERIES d, B N_KEYS d -> B N_QUERIES N_KEYS")
    # dS_ij = P_ij ⊙ (dP_ij - D_i)
    D = O * grad_out 
    D = D.sum(dim = -1)
    dS = P * (dP - D.unsqueeze(-1))
    
    if ctx.is_causal:
        # dQ = dS K / √d
        dQ = einsum(dS * mask, K, "B N_QUERIES N_KEYS, B N_KEYS d -> B N_QUERIES d") / math.sqrt(d)
        # dK = dSᵀ Q / √d
        dK = einsum(dS * mask, Q, "B N_QUERIES N_KEYS, B N_QUERIES d -> B N_KEYS d") / math.sqrt(d)
    else:
        # dQ = dS K / √d
        dQ = einsum(dS, K, "B N_QUERIES N_KEYS, B N_KEYS d -> B N_QUERIES d") / math.sqrt(d)
        # dK = dSᵀ Q / √d
        dK = einsum(dS, Q, "B N_QUERIES N_KEYS, B N_QUERIES d -> B N_KEYS d") / math.sqrt(d)
    return dQ, dK, dV, None

class pytorch_flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # print(Q.shape, K.shape, V.shape, Q.device)
        B, N_QUERIES, d = Q.shape
        _, N_KEYS, _ = K.shape       
        Bq = 16 #128
        Bk = 16 #128 #64 #32 #16
        L = torch.zeros((B, N_QUERIES,), dtype=torch.float32, device=Q.device)
        M = torch.full((B, N_QUERIES), float('-inf'), dtype=torch.float32, device=Q.device)
        O = torch.zeros((B, N_QUERIES, d), dtype=torch.float32, device=Q.device)
        
        # 上下文保存
        ctx.is_causal = is_causal
        # ctx.save_for_backward(Q, K, V, L)
        
        for i in range(0, N_QUERIES, Bq):
            Q_i = Q[:, i:i + Bq, ]
            O_i = O[:, i:i + Bq, ]
            L_i = L[:, i:i + Bq]
            M_i_prev = M[:, i:i + Bq]
            for j in range(0, N_KEYS, Bk):
                # Bk X d
                K_j = K[:, j:j+Bk, ]
                V_j = V[:, j:j+Bk, ]
                # Bq X Bk
                S_ij = einsum(Q_i, K_j, "B Bq d, B Bk d -> B Bq Bk") / math.sqrt(d)
                # Bq
                S_ij_max, _ = torch.max(S_ij, dim=-1)
                M_i_new = torch.maximum(M_i_prev, S_ij_max)
                # Bq X Bk
                # print(S_ij.shape, M_i_new[:,:,None].shape)
                P_ij = torch.exp(S_ij - M_i_new[:,:,None])
                # Bq
                L_i = torch.exp(M_i_prev - M_i_new) * L_i + P_ij.sum(dim=-1)
                # Bq X d
                O_i = einsum(P_ij, V_j, "B Bq Bk, B Bk d -> B Bq d") + einsum(torch.diag_embed(torch.exp(M_i_prev - M_i_new)), O_i, "B Bq1 Bq2, B Bq2 d -> B Bq1 d")
                # update M_i_prev
                M_i_prev = M_i_new

            # Bq X d
            O_i = einsum(torch.diag_embed(1 / L_i), O_i, "B Bq1 Bq2, B Bq2 d -> B Bq1 d")
            # Bq X d
            L_i = M_i_new + torch.log(L_i)
            # update O_i & L_i
            O[:, i:i + Bq, ] = O_i
            L[:, i:i + Bq] = L_i
            M[:, i:i + Bq] = M_i_new

            ctx.save_for_backward(Q, K, V, O, L)

        return O
                
    @staticmethod
    def backward(ctx, grad_out):
        # raise NotImplementedError
        return pytorch_flashattention_backward(ctx, grad_out)
        
        
        
        
        
        


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

import triton
import triton.language as tl

# nvtx
import torch.cuda.nvtx as nvtx

# cs336_basics
from cs336_basics.module import scaled_dot_product_attention_func
from cs336_basics.optim import cross_entropy_func, AdamW
from cs336_systems.pytorch_attention import pytorch_flashattention_backward

# device
from contextlib import nullcontext
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# cudnn
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

@triton.jit
def causal_flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, mask_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_mb, stride_mk, stride_md,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq, 
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    Mask_block_ptr = tl.make_block_ptr(
        mask_ptr,
        shape=(N_QUERIES, N_KEYS),
        strides=(stride_mk, stride_md),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )

    # 临时变量
    L_i = tl.full((Q_TILE_SIZE,), 0, tl.float32)
    M_i = tl.full((Q_TILE_SIZE,), float('-inf'), tl.float32)
    QK_inf = tl.full((Q_TILE_SIZE, K_TILE_SIZE), float('-inf'), tl.float32)
    O_i = tl.full((Q_TILE_SIZE, D), 0, tl.float32)

    # 计算循环
    Q_data = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for k_offset in range(0, N_KEYS, K_TILE_SIZE):
        K_data = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_data = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Mask_data = tl.load(Mask_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # S_ij = einsum(Q_i, K_j, "B Bq d, B Bk d -> B Bq Bk") / math.sqrt(d)
        K_data = tl.trans(K_data)
        S_ij = tl.dot(Q_data, K_data) * scale
        S_ij = tl.where(Mask_data, S_ij, QK_inf)

        # S_ij_max, _ = torch.max(S_ij, dim=-1)
        # M_i_new = torch.maximum(M_i_prev, S_ij_max)
        S_ij_max = tl.max(S_ij, axis = -1)
        M_i_new = tl.maximum(M_i, S_ij_max)

        # P_ij = torch.exp(S_ij - M_i_new[:,:,None])
        P_ij = tl.exp(S_ij - M_i_new[:,None])

        # L_i = torch.exp(M_i_prev - M_i_new) * L_i + P_ij.sum(dim=-1)
        M_exp_sub = tl.exp(M_i - M_i_new)
        L_i = M_exp_sub * L_i + tl.sum(P_ij, axis = -1)

        # O_i = einsum(P_ij, V_j, "B Bq Bk, B Bk d -> B Bq d") + einsum(torch.diag_embed(torch.exp(M_i_prev - M_i_new)), O_i, "B Bq1 Bq2, B Bq2 d -> B Bq1 d")
        O_i = tl.dot(P_ij,  V_data) + M_exp_sub[:, None] * O_i

        # M_i_prev = M_i_new
        M_i = M_i_new

        # Move pointers to next tile
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, K_TILE_SIZE))

    # O_i = einsum(torch.diag_embed(1 / L_i), O_i, "B Bq1 Bq2, B Bq2 d -> B Bq1 d")
    L_i_div = 1 / L_i
    O_i = L_i_div[:, None] * O_i
    L_i = M_i + tl.log(L_i)

    # O[:, i:i + Bq, ] = O_i
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))

    # L[:, i:i + Bq] = L_i
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, mask_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq, 
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    Mask_block_ptr = tl.make_block_ptr(
        mask_ptr,
        shape=(N_QUERIES, N_KEYS),
        strides=(N_KEYS, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0),
    )

    # 临时变量
    L_i = tl.full((Q_TILE_SIZE,), 0, tl.float32)
    M_i = tl.full((Q_TILE_SIZE,), float('-inf'), tl.float32)
    O_i = tl.full((Q_TILE_SIZE, D), 0, tl.float32)

    # 计算循环
    Q_data = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    QK_inf = tl.full((Q_TILE_SIZE, K_TILE_SIZE), -1e6, tl.float32)
    for k_offset in range(0, N_KEYS, K_TILE_SIZE):
        K_data = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_data = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # S_ij = einsum(Q_i, K_j, "B Bq d, B Bk d -> B Bq Bk") / math.sqrt(d)
        K_data = tl.trans(K_data)
        S_ij = tl.dot(Q_data, K_data) * scale

        # mask
        if is_causal:
            Mask_data = tl.load(Mask_block_ptr, boundary_check=(0, 1), padding_option="zero")
            S_ij = tl.where(Mask_data, S_ij, QK_inf)
            Mask_block_ptr = tl.advance(Mask_block_ptr, (0, K_TILE_SIZE))

        # S_ij_max, _ = torch.max(S_ij, dim=-1)
        # M_i_new = torch.maximum(M_i_prev, S_ij_max)
        S_ij_max = tl.max(S_ij, axis = -1)
        M_i_new = tl.maximum(M_i, S_ij_max)

        # P_ij = torch.exp(S_ij - M_i_new[:,:,None])
        P_ij = tl.exp(S_ij - M_i_new[:,None])

        # L_i = torch.exp(M_i_prev - M_i_new) * L_i + P_ij.sum(dim=-1)
        M_exp_sub = tl.exp(M_i - M_i_new)
        L_i = M_exp_sub * L_i + tl.sum(P_ij, axis = -1)

        # O_i = einsum(P_ij, V_j, "B Bq Bk, B Bk d -> B Bq d") + einsum(torch.diag_embed(torch.exp(M_i_prev - M_i_new)), O_i, "B Bq1 Bq2, B Bq2 d -> B Bq1 d")
        O_i = tl.dot(P_ij,  V_data) + M_exp_sub[:, None] * O_i

        # M_i_prev = M_i_new
        M_i = M_i_new

        # Move pointers to next tile
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # O_i = einsum(torch.diag_embed(1 / L_i), O_i, "B Bq1 Bq2, B Bq2 d -> B Bq1 d")
    L_i_div = 1 / L_i
    O_i = L_i_div[:, None] * O_i
    L_i = M_i + tl.log(L_i)

    # O[:, i:i + Bq, ] = O_i
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))

    # L[:, i:i + Bq] = L_i
    tl.store(L_block_ptr, L_i, boundary_check=(0,))
        
    

class trion_flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # print(Q.shape, K.shape, V.shape, Q.device)
        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        Q_TILE_SIZE = 16 #128
        K_TILE_SIZE = 16 #128 #64 #32 #16
        L = torch.zeros((B, N_QUERIES,), dtype=torch.float32, device=Q.device)
        O = torch.zeros((B, N_QUERIES, D), dtype=torch.float32, device=Q.device)

        # if is_causal:
        #     # 应用因果掩码
        mask = torch.tril(torch.ones(N_QUERIES, N_KEYS, dtype=torch.float32, device=Q.device))
        # print(mask)
        
        # 上下文保存
        ctx.is_causal = is_causal
        # ctx.save_for_backward(L)

        # call flash_fwd_kernel
        # if is_causal:
        #     causal_flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)](
        #         Q, K, V, mask,
        #         O, L,
        #         N_QUERIES * D, D, 1,
        #         N_KEYS * D, D, 1,
        #         N_KEYS * D, D, 1,
        #         N_QUERIES * N_KEYS, N_KEYS, 1,
        #         N_QUERIES * D, D, 1,
        #         N_QUERIES, 1, 
        #         N_QUERIES, N_KEYS,
        #         1.0 / math.sqrt(D),
        #         D,
        #         Q_TILE_SIZE,
        #         K_TILE_SIZE
        #     )
        # else:
        flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)](
            Q, K, V, mask,
            O, L,
            N_QUERIES * D, D, 1,
            N_KEYS * D, D, 1,
            N_KEYS * D, D, 1,
            N_QUERIES * D, D, 1,
            N_QUERIES, 1, 
            N_QUERIES, N_KEYS,
            1.0 / math.sqrt(D),
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )
        # print(O[0, :3, :16])
        ctx.save_for_backward(Q, K, V, O, L)
        return O
                
    @staticmethod
    def backward(ctx, grad_out):
        # raise NotImplementedError
        return pytorch_flashattention_backward(ctx, grad_out)
        


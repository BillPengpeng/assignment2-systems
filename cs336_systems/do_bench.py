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

# nvtx
import torch.cuda.nvtx as nvtx

# cs336_basics
from cs336_basics.module import scaled_dot_product_attention_func

# device
from contextlib import nullcontext
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# triton
import triton
import triton.language as tl
from triton.testing import do_bench
from cs336_systems.pytorch_attention import pytorch_flashattention
from cs336_systems.triton_attention import triton_flashattention
# from tests.test_attention import _make_attn_inputs

# cudnn
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

def make_attn_inputs(batch_size, n_queries, n_keys, D, device):
    torch.random.manual_seed(0)
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    do = torch.randn(batch_size, n_queries, D, device=device)
    return q, k, v, do

class self_attention(nn.Module):
    def __init__(self, use_flash_attention=False):
        super().__init__()
        self.attention = triton_flashattention.apply
        self.use_flash_attention = use_flash_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        seq_len = Q.shape[-2]
        if self.use_flash_attention:
            pred = self.attention(Q, K, V)
        else:
            pred = scaled_dot_product_attention_func(Q, K, V, mask)
        return pred

def benchmark(fn: Callable, num_warmups: int = 5, num_trials: int = 3):
    time_ms = do_bench(
        fn,             # 要测试的函数
        quantiles=None, # 是否返回分位数
        warmup=num_warmups,  # 预热次数
        rep=num_trials,      # 正式测量次数
        grad_to_none=None    # 梯度张量处理
    )
    return time_ms

def run_forward(model, batch_size, n_queries, n_keys, D) -> Callable:
    q, k, v, _do = make_attn_inputs(batch_size, n_queries, n_keys, D, device)
    mask = torch.tril(torch.ones(n_queries, n_keys, dtype=torch.float32, device=device))
    def run():
        out = model(q, k, v, mask)
    return run

def run_backward(model, batch_size, n_queries, n_keys, D) -> Callable:
    q, k, v, _do = make_attn_inputs(batch_size, n_queries, n_keys, D, device)
    mask = torch.tril(torch.ones(n_queries, n_keys, dtype=torch.float32, device=device))
    def run():
        out = model(q, k, v, mask)
        out.backward(_do)
    return run

def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 1024 #8192 #16384
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    # print(q.shape, k.shape, v.shape)
    flash = triton_flashattention.apply
    # flash = torch.compile(triton_flashattention.apply)
    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()
        
    results = triton.testing.do_bench(flash_forward_backward, rep=100, warmup=10)
    print("test_timing_flash_forward_backward:", results)

if __name__ == "__main__":
    # test_timing_flash_forward_backward
    test_timing_flash_forward_backward()

    # flash_benchmarking 
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] 
    batch_size = 1
    num_warmups = 10
    num_steps  = 100
    data = {
        "seq_len": list(),
        "d_model": list(),
        "forward_time": list(),
        "forward_backward_time": list()
    }
    use_flash_attention = True
    use_mixed_precision = False
    if use_mixed_precision:
        context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        context_manager = nullcontext()
    
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            data['seq_len'].append(seq_len)
            data['d_model'].append(d_model)
            model = self_attention(use_flash_attention)
            model = torch.compile(model)
            with context_manager:
                try:
                    manual_time = benchmark(
                        run_forward(model, batch_size, seq_len, seq_len, d_model),
                        num_warmups = num_warmups,
                        num_trials = num_steps
                    )
                    data["forward_time"].append(manual_time)
                except:
                    data["forward_time"].append(None)
    
                try:
                    manual_time = benchmark(
                        run_backward(model, batch_size, seq_len, seq_len, d_model),
                        num_warmups = num_warmups,
                        num_trials = num_steps
                    )
                    data["forward_backward_time"].append(manual_time)
                except:
                    data["forward_backward_time"].append(None)

    df = pd.DataFrame(data)
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)

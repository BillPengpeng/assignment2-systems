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

class self_attention(nn.Module):
    def __init__(self, d_model: int, 
                 max_seq_len: int | None=None,  
                 device: torch.device | None=None, 
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.d_model = d_model
        self.o_weight = nn.Parameter(torch.empty(self.d_model, self.d_model, dtype=dtype, device=device))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        seq_len = Q.shape[-2]
        pred = scaled_dot_product_attention_func(Q, K, V)
        # pred = einsum(self.o_weight, pred, "out_dim in_dim, ... seq_len in_dim -> ... seq_len out_dim")
        return pred
        # return (mid, pred)

def benchmark(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 3, use_memory_profiling: bool = False):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    nvtx.range_push(f"warmups")
    for _ in range(num_warmups):
        run()
    nvtx.range_pop()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # start profiling after warmup iterations
    # torch.cuda.cudart().cudaProfilerStart()

    # # Start recording memory history.
    # if use_memory_profiling:
    #     torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Time it for real now!
    times = dict()
    for trial in range(num_trials):  # Do it multiple times to capture variance
        nvtx.range_push(f"step_{trial}")
        rst = run()  # Actually perform computation
        nvtx.range_pop()
        for keyname in rst.keys():
            if keyname not in times.keys():
                times[keyname] = list()
            times[keyname].append(rst[keyname])

    # 结束分析
    # torch.cuda.cudart().cudaProfilerStop()

    # # Save a pickle file to be loaded by PyTorch's online tool.
    # if use_memory_profiling:
    #     torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
    # # Stop recording history.
    # if use_memory_profiling:
    #     torch.cuda.memory._record_memory_history(enabled=None)

    # print(times)
    times_key_list = list(times.keys())
    for keyname in times_key_list:
        times[keyname + '_mean'] = np.mean(times[keyname])
        times[keyname + '_var']  = np.var(times[keyname])
    return times

def parse_args():
    parser = argparse.ArgumentParser(description='pytorch_attention')
    parser.add_argument('--auto', type=int, default=0, help='batch eval model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--seq_len', type=int, default=256, help='seq_len')
    parser.add_argument('--d_model', type=int, default=16, help='d_model')
    parser.add_argument('--num_warmups', type=int, default=5, help='num_warmups')
    parser.add_argument('--num_steps', type=int, default=10, help='rope_theta')
    parser.add_argument('--only_forward', type=bool, default=False, help='only_forward')
    parser.add_argument('--use_mixed_precision', type=bool, default=False, help='use_mixed_precision')
    parser.add_argument('--use_memory_profiling', type=bool, default=False, help='memory_profiling')
    return parser.parse_args()

def run_model(
    model: nn.Module,
    batch_size: int = 4,
    seq_len: int = 256,
    d_model: int = 512,
    only_forward: bool = True,
    use_mixed_precision: bool = False,
    use_memory_profiling: bool = False
) -> Callable:
    times = dict()
    def run():
        # data
        Q = torch.rand(batch_size, seq_len, d_model, dtype=torch.float32, device=device, requires_grad=True)
        K = torch.rand(batch_size, seq_len, d_model, dtype=torch.float32, device=device, requires_grad=True)
        V = torch.rand(batch_size, seq_len, d_model, dtype=torch.float32, device=device, requires_grad=True)

        if use_mixed_precision:
            context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            context_manager = nullcontext()

        # forward
        with context_manager:
            # pre_forward_allocated = torch.cuda.memory_allocated()
            # pre_forward_allocated = torch.cuda.max_memory_allocated()
            start_time = default_timer() #time.time() 
            with nvtx.range("forward"):
                pred = model(Q, K, V)
                # pred = scaled_dot_product_attention_func(Q, K, V)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
            end_time = default_timer() #time.time() 
            times['forward'] = (end_time - start_time) * 1000

            # post_forward_allocated = torch.cuda.max_memory_allocated()
            # times['allocated'] = (post_forward_allocated - pre_forward_allocated) / (1024**2)
    
            if (not only_forward) and (not use_memory_profiling):
                # backward
                start_time = default_timer() #time.time() 
                with nvtx.range("backward"):
                    loss = pred.mean()  
                    loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
                end_time = default_timer() #time.time() 
                times['backward'] = (end_time - start_time) * 1000 
            
        return times
        
    return run

def run_benchmark(
    batch_size: int = 4,
    seq_len: int = 256,
    d_model: int = 512,
    num_warmups: int = 5,
    num_steps: int = 10,
    only_forward: bool = False,
    use_mixed_precision: bool = False,
    use_memory_profiling: bool = False
):
    # # Start recording memory history.
    # if use_memory_profiling:
    #     torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # benchmark
    model = self_attention(d_model, max_seq_len=seq_len, dtype=torch.float32, device=device)

    # torch.compile
    model = torch.compile(model)
    manual_time = benchmark(
        "pytorch_attention", 
        run_model(model, batch_size, seq_len, d_model, only_forward, use_mixed_precision, use_memory_profiling=use_memory_profiling), 
        num_warmups=num_warmups, 
        num_trials=num_steps,
        use_memory_profiling=use_memory_profiling
    )

    # # Save a pickle file to be loaded by PyTorch's online tool.
    # if use_memory_profiling:
    #     torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
    # # Stop recording history.
    # if use_memory_profiling:
    #     torch.cuda.memory._record_memory_history(enabled=None)
    return manual_time


if __name__ == "__main__":
    args = parse_args()
    # Start recording memory history.
    if args.use_memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    if args.auto:
        batch_size = 8
        d_model_list = [16, 32, 64, 128]
        seq_len_list = [256, 1024, 4096, 8192, 16384]
        # d_model_list = [16]
        # seq_len_list = [256, 1024] #, 4096, 8192] #, 16384]
        num_warmups = 4
        num_steps  = 100
        only_forward = False #True #
        use_mixed_precision = False
        data = {
            "batch_size": list(),
            "seq_len": list(),
            "d_model": list()
        }
        req_key_list = list(data.keys())
        for d_model in d_model_list:
            for seq_len in seq_len_list:
                data['batch_size'].append(batch_size)
                data['seq_len'].append(seq_len)
                data['d_model'].append(d_model)
                try:
                    manual_time = run_benchmark(
                        batch_size = batch_size,
                        seq_len = seq_len,
                        d_model = d_model,
                        num_warmups = num_warmups,
                        num_steps = num_steps,
                        only_forward = only_forward,
                        use_mixed_precision = use_mixed_precision,
                        use_memory_profiling = args.use_memory_profiling
                    )
                    for key_name in manual_time.keys():
                        if '_' not in key_name:
                            continue
                        if key_name not in data.keys():
                            data[key_name] = list()
                        data[key_name].append(manual_time[key_name])
                except:
                    for key_name in data.keys():
                        if key_name in req_key_list:
                            continue
                        data[key_name].append(None)

        df = pd.DataFrame(data)
        markdown_table = df.to_markdown(index=False)
        print(markdown_table)

    else:    
        manual_time = run_benchmark(
            batch_size = args.batch_size,
            seq_len = args.seq_len,
            d_model = args.d_model,
            num_warmups = args.num_warmups,
            num_steps = args.num_steps,
            only_forward = args.only_forward,
            use_mixed_precision = args.use_mixed_precision,
            use_memory_profiling = args.use_memory_profiling
        )
        print(manual_time)

    # Save a pickle file to be loaded by PyTorch's online tool.
    if args.use_memory_profiling:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
    # Stop recording history.
    if args.use_memory_profiling:
        torch.cuda.memory._record_memory_history(enabled=None)
    

    

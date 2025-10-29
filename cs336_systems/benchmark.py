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
from cs336_basics.module import transformer_lm
from cs336_basics.optim import cross_entropy_func, AdamW

# device
from contextlib import nullcontext
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# cudnn
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

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
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--auto', type=int, default=0, help='batch eval model')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocab_size')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--seq_len', type=int, default=256, help='seq_len')
    parser.add_argument('--d_model', type=int, default=1280, help='d_model')
    parser.add_argument('--d_ff', type=int, default=5120, help='d_ff')
    parser.add_argument('--num_layers', type=int, default=36, help='num_layers')
    parser.add_argument('--num_heads', type=int, default=20, help='num_heads')
    parser.add_argument('--rope_theta', type=int, default=10000, help='rope_theta')
    parser.add_argument('--num_warmups', type=int, default=5, help='num_warmups')
    parser.add_argument('--num_steps', type=int, default=10, help='rope_theta')
    parser.add_argument('--only_forward', type=bool, default=True, help='only_forward')
    parser.add_argument('--use_mixed_precision', type=bool, default=False, help='use_mixed_precision')
    parser.add_argument('--use_memory_profiling', type=bool, default=False, help='memory_profiling')
    return parser.parse_args()

def build_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float
):
    model = transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
    return model

def run_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    batch_size: int,
    context_length: int,
    only_forward: bool,
    use_mixed_precision: bool
) -> Callable:
    times = dict()
    def run():
        # data
        data = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
        labels = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)

        if use_mixed_precision:
            context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            context_manager = nullcontext()

        # forward
        with context_manager:
            start_time = default_timer() #time.time() 
            with nvtx.range("forward"):
                pred = model(data) 
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
            end_time = default_timer() #time.time() 
            times['forward'] = (end_time - start_time) * 1000
    
            if not only_forward:
                # backward
                start_time = default_timer() #time.time() 
                with nvtx.range("backward"):
                    loss = cross_entropy_func(pred, labels)     
                    loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
                end_time = default_timer() #time.time() 
                times['backward'] = (end_time - start_time) * 1000 
    
                # optimizer
                start_time = default_timer() #time.time() 
                with nvtx.range("optimizer_step"):
                    optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
                end_time = default_timer() #time.time() 
                times['optimizer'] = (end_time - start_time) * 1000 
            
        return times
        
    return run

def run_benchmark(
    vocab_size: int = 10000,
    batch_size: int = 4,
    seq_len: int = 256,
    d_model: int = 512,
    d_ff: int = 1344,
    num_layers: int = 4,
    num_heads: int = 16,
    rope_theta: int = 10000,
    num_warmups: int = 5,
    num_steps: int = 10,
    only_forward: bool = False,
    use_mixed_precision: bool = False,
    use_memory_profiling: bool = False
):
    # Start recording memory history.
    if use_memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # model
    model = build_model(
        vocab_size,
        seq_len,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta
    )

    # optimizer
    optimizer = AdamW(model.parameters())
    
    # benchmark
    manual_time = benchmark(
        "transformer_lm", 
        run_model(model, optimizer, vocab_size, batch_size, seq_len, only_forward, use_mixed_precision), 
        num_warmups=num_warmups, 
        num_trials=num_steps,
        use_memory_profiling=use_memory_profiling
    )

    # Save a pickle file to be loaded by PyTorch's online tool.
    if use_memory_profiling:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
    # Stop recording history.
    if use_memory_profiling:
        torch.cuda.memory._record_memory_history(enabled=None)
    return manual_time


if __name__ == "__main__":
    args = parse_args()

    if args.auto:
        vocab_size = 10000
        batch_size = 4
        seq_len    = 256
        rope_theta  = 10000
        num_warmups = 4
        num_steps  = 10
        only_forward = False #True
        use_mixed_precision = False #True # False
        use_memory_profiling = False
        data = {
            "size": list(),
            "batch_size": list(),
            "seq_len": list(),
            "d_model": list(),
            "d_ff": list(),
            "num_layers": list(),
            "num_heads": list()
        }
        req_key_list = list(data.keys())
        small  = {"size":"small", "d_model":768,  "d_ff":3072,  "num_layers":12, "num_heads":12}
        medium = {"size":"medium", "d_model":1024, "d_ff":4096,  "num_layers":24, "num_heads":16}
        large  = {"size":"large", "d_model":1280, "d_ff":5120,  "num_layers":36, "num_heads":20}
        xl     = {"size":"xl", "d_model":1600, "d_ff":6400,  "num_layers":48, "num_heads":25}
        L_27B  = {"size":"2.7B", "d_model":2560, "d_ff":10240, "num_layers":32, "num_heads":32}
        for model_dict in [small, medium, large, xl, L_27B]:
            for seq_len in [128, 256, 512, 1024]:
                data['size'].append(model_dict['size'])
                data['batch_size'].append(batch_size)
                data['seq_len'].append(seq_len)
                data['d_model'].append(model_dict['d_model'])
                data['d_ff'].append(model_dict['d_ff'])
                data['num_layers'].append(model_dict['num_layers'])
                data['num_heads'].append(model_dict['num_heads'])
                try:
                    manual_time = run_benchmark(
                        vocab_size = vocab_size,
                        batch_size = batch_size,
                        seq_len = seq_len,
                        d_model = model_dict['d_model'],
                        d_ff = model_dict['d_ff'],
                        num_layers = model_dict['num_layers'],
                        num_heads = model_dict['num_heads'],
                        rope_theta = rope_theta,
                        num_warmups = num_warmups,
                        num_steps = num_steps,
                        only_forward = only_forward,
                        use_mixed_precision = use_mixed_precision,
                        use_memory_profiling = use_memory_profiling
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
            vocab_size = args.vocab_size,
            batch_size = args.batch_size,
            seq_len = args.seq_len,
            d_model = args.d_model,
            d_ff = args.d_ff,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            rope_theta = args.rope_theta,
            num_warmups = args.num_warmups,
            num_steps = args.num_steps,
            only_forward = args.only_forward,
            use_mixed_precision = args.use_mixed_precision,
            use_memory_profiling = args.use_memory_profiling
        )
        print(manual_time)
    

    

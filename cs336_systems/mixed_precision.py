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
device_str = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# cudnn
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

# mixed_precision_accumulation
def mixed_precision_accumulation():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)
    
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

# ToyModel
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False, device=device)
        self.ln = nn.LayerNorm(10, device=device)
        self.fc2 = nn.Linear(10, out_features, bias=False, device=device)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.ln(x2)
        x4 = self.fc2(x3)
        return (x1, x2, x3, x4)

def run_toy_model():
    model = ToyModel(256, 256, device=device)
    data = torch.rand(1, 256, dtype=torch.float32, device=device)
    # with torch.autocast(device="cuda", dtype=torch.float16):
    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x1, x2, x3, x4 = model(data) 
        loss = x4.mean()
        loss.backward()

    # parameters
    # for p in model.parameters():
    #     import pdb;pdb.set_trace()
    print("fc1.weight:", model.fc1.weight.dtype)
    print("fc1.weight.grad:", model.fc1.weight.grad.dtype)
    print("fc1_output:", x1.dtype)
    
    print("ln.weight:", model.ln.weight.dtype)
    print("ln.weight.grad:", model.ln.weight.grad.dtype)
    print("ln_output:", x3.dtype)

    print("fc2.weight:", model.fc2.weight.dtype)
    print("fc2.weight.grad:", model.fc2.weight.grad.dtype)
    print("fc2_output:", x4.dtype)
    print("loss:", loss.dtype)
    
    # print("ln.weight:", model.fc1.weight.type)
    # print("fc1.weight:", model.fc1.weight.type)
    # print("fc1.weight:", model.fc1.weight.type)
    

if __name__ == "__main__":
    # mixed_precision_accumulation()
    run_toy_model()
    

    

import torch
import torch.nn as nn
from dataclasses import dataclass

from baseline import Baseline
from ssa import SSA
from slsa import SLSA
from lsa import LSA
from vsa import VSA
from vlsa import VLSA

@dataclass
class GPTConfig:
    block_size: int = 512  # Increased from 254
    vocab_size: int = 50257  # Kept the same
    n_layer: int = 12  # Increased from 8
    n_head: int = 12  # Increased from 8
    n_embd: int = 768  # Increased from 256

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, head_class):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = head_class(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config, head_class):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, head_class) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

def calculate_model_params(head_class):
    config = GPTConfig()
    model = GPT(config, head_class)
    return count_parameters(model)

if __name__ == "__main__":
    head_classes = [Baseline, SSA, SLSA, LSA, VSA, VLSA]
    head_names = ["Baseline", "SSA", "SLSA", "LSA", "VSA", "VLSA"]

    for name, head_class in zip(head_names, head_classes):
        num_params = calculate_model_params(head_class)
        print(f"{name} total parameters: {num_params:,}")

    baseline_params = calculate_model_params(Baseline)
    for name, head_class in zip(head_names[1:], head_classes[1:]):
        num_params = calculate_model_params(head_class)
        diff_percentage = ((num_params - baseline_params) / baseline_params) * 100
        print(f"{name} difference from Baseline: {diff_percentage:.2f}%")
import os
import argparse
import torch
import torch.nn as nn

import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
from heads.baseline import Baseline
from heads.ssa import SSA

# head_class = SSA
head_class = Baseline

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # type: ignore
    device = "mps"
print(f"using device: {device}")


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = head_class(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 512


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # type: ignore

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (T, n_embd) # type: ignore
        tok_emb = self.transformer.wte(
            idx
        )  # token embeddings of shape (B, T, n_embd) # type: ignore
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h: # type: ignore
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        return cls(GPTConfig())


def sample_from_model(
    model,
    prompt,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    n_samples=1,
):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    for _ in range(n_samples):
        context = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
        generated_tokens = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = model(context)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token.item())
            context = torch.cat((context, next_token), dim=1)

        y = enc.decode(generated_tokens)
        print(f"{prompt}{y}")
        print("---------------\n")

    return None


prompt = "In this test,"


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained GPT model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model",
        # default="./log/baseline_log.txt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=f"{prompt}",
        help="Prompt to start generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling (optional)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples to generate"
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # Load the saved model
    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = checkpoint["config"]
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])

    # Generate text
    generated_text = sample_from_model(
        model,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.n_samples,
    )


if __name__ == "__main__":
    main()
    # python .\sample.py --model_path=./log/model_simple_attention.pt
    # python .\sample.py --model-path=./log/runs/baseline_20240804-082315/baseline_20240804-082315.pt
    # python .\sample.py --model_path=./log/runs/simple_attention_20240731-154846/simple_attention_20240731-154846.pt
    # python .\sample.py --model_path=./log/runs/baseline_20240804-082315/baseline_20240804-082315.pt

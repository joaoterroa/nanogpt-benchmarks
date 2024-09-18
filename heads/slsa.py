import torch.nn as nn
from torch.nn import functional as F
import torch
import math


class SLSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.w_u = nn.Linear(config.n_embd, config.n_embd)
        self.w_v = nn.Linear(config.n_embd, config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore
        self.l1 = nn.Linear(self.head_size, self.head_size)
        self.l2 = nn.Linear(self.head_size, self.head_size)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        (B, T, C) = x.size()

        # config_dict = {"B": B, "T": T, "C": C, "nh": self.n_head, "hs": self.head_size}

        u = self.w_u(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        x_transpose = (
            x.view(B, T, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1)
        )
        att = torch.matmul(
            u, x_transpose
        )  # (B, nh, T, hs) x (B, nh, hs, T) = (B, nh, T, T)
        att = att * (1.0 / math.sqrt(x.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        att = F.softmax(att, dim=-1)

        v = self.w_v(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        l1 = self.l1(v)
        l2 = torch.sigmoid(self.l2(v))
        v = v + l1
        v = v * l2

        y = att @ v  # (B, n_head, T, T) x (B, n_head, T, HS) -> (B, n_head, T, HS)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

    # def forward(self, x):
    #     (
    #         B,
    #         T,
    #         C,
    #     ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
    #     # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    #     # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    #     # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    #     u = torch.matmul(
    #         x.unsqueeze(1), self.unique.weight.reshape(1, self.n_head, C, C)
    #     )
    #     # u = x @ self.unique.weight.view(B, self.n_head, C, C)
    #     v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    #     att = torch.matmul(u, x.unsqueeze(1).transpose(2, 3))
    #     att = att * (1.0 / math.sqrt(x.size(-1)))
    #     att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
    #     att = F.softmax(att, dim=-1)

    #     l1 = self.l1(v)
    #     l2 = torch.sigmoid(self.l2(v))
    #     v = v + l1
    #     v = v * l2
    #     y = att @ v  # (B, n_head, T, T) x (B, n_head, T, HS) -> (B, n_head, T, HS)

    #     y = att @ v
    #     y = (
    #         y.transpose(1, 2).contiguous().view(B, T, C)
    #     )  # re-assemble all head outputs side by side
    #     # output projection
    #     y = self.c_proj(y)
    #     return y

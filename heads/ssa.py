import torch.nn as nn
from torch.nn import functional as F
import torch
import math


def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None  # If the value is not found, return None


def create_string_from_shape(tensor, dictionary):
    s = "["
    for i in range(len(tensor.shape)):
        shape_at_index = tensor.shape[i]
        key = get_key_by_value(dictionary, shape_at_index)

        s += f"{key}, "
    s = s[:-2] + "]"
    return s


class SSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.hs = config.n_embd // config.n_head

        self.w_u = nn.Linear(config.n_embd, config.n_embd)
        self.w_v = nn.Linear(config.n_embd, config.n_embd)
        self.helper = nn.Linear(
            config.n_embd, self.hs
        )  # TODO TREINAR DE NOVO SEM ESTA LINHA

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        (B, T, C) = x.size()

        # config_dict = {"B": B, "T": T, "C": C, "nh": self.n_head, "hs": self.hs}

        u = self.w_u(x).view(B, T, self.n_head, self.hs).transpose(1, 2)
        x_transpose = (
            x.view(B, T, self.n_head, self.hs).transpose(1, 2).transpose(-2, -1)
        )
        att = torch.matmul(
            u, x_transpose
        )  # (B, nh, T, hs) x (B, nh, hs, T) = (B, nh, T, T)
        att = att * (1.0 / math.sqrt(x.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        att = F.softmax(att, dim=-1)

        v = self.w_v(x).view(B, T, self.n_head, self.hs).transpose(1, 2)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)

        return y


if __name__ == "__main__":
    pass

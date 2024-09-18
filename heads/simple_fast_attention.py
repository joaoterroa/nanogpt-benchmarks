import torch.nn as nn
from torch.nn import functional as F
import torch
import math


# def get_key_by_value(dictionary, target_value):
#     for key, value in dictionary.items():
#         if value == target_value:
#             return key
#     return None  # If the value is not found, return None


# def create_string_from_shape(tensor, dictionary):
#     s = "["
#     for i in range(len(tensor.shape)):
#         shape_at_index = tensor.shape[i]
#         key = get_key_by_value(dictionary, shape_at_index)

#         s += f"{key}, "
#     s = s[:-2] + "]"
#     return s


class SimpleFastAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.hs = config.n_embd // config.n_head

        self.w_u = nn.Linear(config.n_embd, config.n_embd)
        self.helper = nn.Linear(config.n_embd, self.hs)

        self.w_v = nn.Linear(config.n_embd, config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.hs, self.hs)).view(1, 1, self.hs, self.hs),
        )

    def forward(self, x):
        (B, T, C) = x.size()

        # config_dict = {"B": B, "T": T, "C": C, "nh": self.n_head, "hs": self.hs}

        # X . (W_u . X^T) . (X . W_v) = X . U . V
        u = torch.matmul(self.w_u.weight, x.transpose(1, 2))
        u = u.view(B, self.n_head, self.hs, T)
        v = self.w_v(x)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)

        att = torch.matmul(u, v)  # (B, nh, hs, T) x (B, nh, T, hs) = (B, nh, hs, hs)

        att = att * (1.0 / math.sqrt(x.size(-1)))
        att = att.masked_fill(self.bias[:, :, : self.hs, : self.hs] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, self.helper.weight).view(B, C, C)

        y = x @ att  # (B, T, C) x (B, C, C) = (B, T, C)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


if __name__ == "__main__":

    class Config:
        def __init__(self, n_embd, n_head, block_size):
            self.n_embd = n_embd
            self.n_head = n_head
            self.block_size = block_size

    # Define a dummy configuration with arbitrary values
    dummy_config = Config(n_embd=768, n_head=16, block_size=1024)
    C = dummy_config.n_embd
    nh = dummy_config.n_head
    T = dummy_config.block_size
    hs = C / nh
    # Initialize the model
    model = SimpleFastAttention(dummy_config)
    print(model.forward(torch.randn(1, T, C)).shape)
    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    # for k, v in model.named_parameters():
    #     print(f"{k}: {v.numel()}")
    print(f"Total number of parameters: {num_params}")
    # num_params = (nh * C * hs) + (C * C * nh)
    # increase = (((num_params) - (3 * C * C)) / (3 * C * C)) * 100
    # print(f"Percentage increase: {increase:.2f}")
    # print(int(num_params * (3 / (nh + 1))))

import torch


def calculate_perplexity(loss, num_steps):
    perplexity = torch.exp(loss / num_steps)
    return perplexity.item()

"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, concatenate_datasets, Dataset  # pip install datasets

from tqdm import tqdm  # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"  # remote dataset name
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards
# name = "edufineweb"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# download the dataset
# fw = load_dataset("nampdn-ai/tiny-textbooks", split="train")
# fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# cache_dir = os.path.expanduser(
#     "C:/Users/Joao.DESKTOP-TD93RS5/.cache/huggingface/datasets"
# )
# dataset_path = os.path.join(cache_dir, "HuggingFaceFW___fineweb-edu", remote_name)

# full_path = os.path.join(dataset_path, "0.0.0/6a3a8126a53ad5fc")


# arrow_files = [f for f in os.listdir(full_path) if f.endswith(".arrow")]
# fw = concatenate_datasets(
#     [
#         Dataset.from_file(os.path.join(full_path, arrow_file))
#         for arrow_file in arrow_files
#     ]
# )

# # Function to load all Arrow files in a directory
# def load_arrow_files(directory):
#     arrow_files = [f for f in os.listdir(directory) if f.endswith(".arrow")]
#     datasets = []
#     for file in arrow_files:
#         dataset = Dataset.from_file(os.path.join(directory, file))
#         datasets.append(dataset)
#     return datasets


# # Load the dataset from the local cache
# fw_datasets = load_arrow_files(full_path)

# # Combine all datasets
# fw = Dataset.from_datasets(fw_datasets)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2) # type: ignore
    print(f"{nprocs} processes")
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder) # type: ignore
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()

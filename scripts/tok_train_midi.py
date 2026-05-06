"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer_midi import REMIBPETokenizer 
from nanochat.common import get_base_dir
# from nanochat.dataset_midi import get_midi_string
from pathlib import Path

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-notes', type=int, default=2_000_000_000, help='Maximum notes to train on (default: 2B)')
# parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_notes: {args.max_notes:,}")
# print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_midi")



# -----------------------------------------------------------------------------
# Text iterator

# def text_iterator():
#     """
#     1) Flatten the batches into a single iterator
#     2) Crop every document to args.doc_cap characters
#     3) Break when we've seen args.max_chars characters
#     """
#     nnotes = 0
#     # for batch in parquets_iter_batched(split="train"):
#     #     for doc in batch:
#     #         doc_text = doc
#     #         if len(doc_text) > args.doc_cap:
#     #             doc_text = doc_text[:args.doc_cap]
#     #         nchars += len(doc_text)
#     #         yield doc_text
#     #         if nchars > args.max_chars:
#     #             return
            
#     for doc in get_midi_string(split="train"):
#         # for doc in batch:
#         doc_text = doc
#         if len(doc_text) > args.doc_cap:
#             doc_text = doc_text[:args.doc_cap]
#         nnotes += len(doc_text)
#         yield doc_text
#         if nnotes > args.max_chars:
#             return
        
#     files_paths = [Path(os.getcwd(),'tmp_data') / f for f in os.listdir('tmp_data') if f.endswith('.mid')]
# text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
# files_paths = [Path(os.getcwd(),'tmp_data') / f for f in os.listdir('tmp_data') if f.endswith('.mid')]
files_paths = [Path(DATA_DIR) / f for f in os.listdir(DATA_DIR) if f.endswith('.mid')]
t0 = time.time()
# tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
tokenizer = REMIBPETokenizer.train_from_iterator(files_paths, args.vocab_size)
# tokenizer = REMIBPETokenizer.train_from_iterator(files_paths[:10], args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
# test_text = """Hello world! This is a test.
# Numbers: 123, 4567, 89
# Contractions: I'm, you're, it's
# Special chars: @#$%^&*()
# Unicode: 你好世界 🌍"""
# encoded = tokenizer.encode(test_text)
# decoded = tokenizer.decode(encoded)
# assert decoded == test_text

score = tokenizer.enc.encode(files_paths[0])
ids = score.ids
reconstructed_score = tokenizer.enc.decode(ids)
reconstructed_tokens = tokenizer.enc._score_to_tokens(reconstructed_score).tokens
assert score.tokens == reconstructed_tokens

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if len(token_str) == 1 and token_str[0] in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        joined_str = "".join(token_str)
        id_bytes = len(joined_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])

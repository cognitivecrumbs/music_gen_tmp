"""
nanochat/dataloader.py  –  REPLACE the original
================================================
The real dataloader.py:
  - imports list_parquet_files from nanochat.dataset  (✓ now exists)
  - yields (inputs, targets, state_dict) triples from the _with_state variant
  - the non-state variant yields (inputs, targets) pairs

base_train.py usage:
  train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
      tokenizer, B, T, split="train", device=device,
      resume_state_dict=dataloader_resume_state_dict)
  x, y, dataloader_state_dict = next(train_loader)

  build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
      tokenizer, B, T, split="val", device=device)
  # val_loader used with next() inside evaluate_bpb

Our replacement keeps all signatures identical, reads pre-tokenised
uint16 .bin shards, and emits a state_dict with the same keys base_train.py
logs: {"pq_idx": ..., "rg_idx": ..., "epoch": ...}
(named after parquet concepts but now mean shard/position/epoch).
"""

import torch
import random
from nanochat.common import get_dist_info
from nanochat.dataset import list_shard_files   # our .bin shard lister

import numpy as np


# ---------------------------------------------------------------------------
# Internal shard stream
# ---------------------------------------------------------------------------

class _ShardStream:
    def __init__(self, shard_paths, rank, world_size, resume_state_dict=None):
        self.paths      = shard_paths
        self.rank       = rank
        self.world_size = world_size
        self.epoch      = 1

        if resume_state_dict is not None:
            self._shard_idx = resume_state_dict.get("pq_idx", 0)
            self._pos       = resume_state_dict.get("rg_idx", 0)
            self.epoch      = resume_state_dict.get("epoch", 1)
        else:
            self._shard_idx = 0
            self._pos       = rank  # stride by rank

        self._tokens = self._load(self._shard_idx)

    def _load(self, idx):
        path = self.paths[idx % len(self.paths)]
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))

    def state_dict(self):
        return {"pq_idx": self._shard_idx, "rg_idx": self._pos, "epoch": self.epoch}

    def read(self, n):
        out = []
        remaining = n
        while remaining > 0:
            available = len(self._tokens) - self._pos
            take = min(available, remaining)
            out.append(self._tokens[self._pos : self._pos + take])
            self._pos += take
            remaining -= take
            if self._pos >= len(self._tokens):
                self._shard_idx += 1
                if self._shard_idx >= len(self.paths):
                    self._shard_idx = 0
                    self.epoch += 1
                self._tokens = self._load(self._shard_idx)
                self._pos = self.rank
        return torch.cat(out)


def _make_stream(split, resume_state_dict=None):
    _, ddp_rank, _, ddp_world_size = get_dist_info()
    all_shards = list_shard_files()
    random.seed(42)
    shards = list(all_shards)
    random.shuffle(shards)
    n_val = max(1, len(shards) // 20)
    shards = shards[:n_val] if split == "val" else shards[n_val:]
    assert shards, f"No shards for split='{split}'"
    return _ShardStream(shards, rank=ddp_rank, world_size=ddp_world_size,
                        resume_state_dict=resume_state_dict)


# ---------------------------------------------------------------------------
# Public API — exact same signatures as the real nanochat dataloader
# ---------------------------------------------------------------------------

def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
):
    """
    Yields (inputs, targets, state_dict) triples — matching real nanochat.
    base_train.py: x, y, dataloader_state_dict = next(train_loader)
    """
    stream = _make_stream(split, resume_state_dict)
    dev = torch.device(device) if isinstance(device, str) else device

    use_cuda = (dev.type == "cuda")
    cpu_buf = torch.empty(2 * B * T, dtype=torch.long,
                          pin_memory=use_cuda)
    gpu_buf = torch.empty(2 * B * T, dtype=torch.long, device=dev)
    cpu_x = cpu_buf[:B*T].view(B, T)
    cpu_y = cpu_buf[B*T:].view(B, T)
    x = gpu_buf[:B*T].view(B, T)
    y = gpu_buf[B*T:].view(B, T)

    while True:
        flat = stream.read(B * T + 1)
        cpu_x.copy_(flat[:-1].view(B, T))
        cpu_y.copy_(flat[1: ].view(B, T))
        gpu_buf.copy_(cpu_buf, non_blocking=use_cuda)
        yield x, y, stream.state_dict()


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Yields (inputs, targets) pairs — omits state_dict."""
    for inputs, targets, _ in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

"""
nanochat/dataset.py  –  REPLACE the original
=============================================
The real dataloader.py imports:
    from nanochat.dataset import list_parquet_files

We keep that exact name but make it return our .bin shard paths,
so dataloader.py's import succeeds. Our replacement dataloader (02b)
never actually calls list_parquet_files() — it uses list_shard_files()
directly — but the name must exist for the import not to crash.

Run once before training:
    python -m nanochat.dataset --midi-dir /path/to/midi
"""

import argparse
import os
from pathlib import Path
from typing import List
import pickle
import numpy as np

# from nanochat.tokenizer import midi_to_tokens
from nanochat.tokenizer_pre_update import midi_to_tokens, tokens_to_midi

DEFAULT_CACHE = os.path.expanduser("~/.cache/nanochat")
SHARD_SIZE    = 1_000_000

DATA_DIR = os.path.join(DEFAULT_CACHE, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)


def list_shard_files(data_dir: str = DATA_DIR) -> List[str]:
    """Return sorted list of pre-tokenised uint16 .bin shard paths."""
    paths = sorted(Path(data_dir).glob("shard_*.bin"))
    if not paths:
        raise RuntimeError(
            f"No shard_*.bin files found in {data_dir}\n"
            f"Run:  python -m nanochat.dataset --midi-dir /path/to/your/midi"
        )
    return [str(p) for p in paths]


def list_parquet_files(warn_on_legacy: bool = False) -> List[str]:
    """
    nanochat's dataloader.py imports this name.
    In nanomusic we return .bin shard paths instead of .parquet paths.
    Our replacement dataloader never calls this directly, but the import
    must not raise ImportError.
    """
    return list_shard_files()


def parquets_iter_batched(*args, **kwargs):
    raise NotImplementedError(
        "parquets_iter_batched is not used in nanomusic. "
        "Tokens are pre-computed .bin shards — see 02b_dataloader.py."
    )


def tokenize_directory(midi_dir: str, out_dir: str = DATA_DIR, shard_size: int = SHARD_SIZE):
    midi_dir = Path(midi_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(midi_dir.rglob("*.mid")) + sorted(midi_dir.rglob("*.midi"))
    print(f"Found {len(files)} MIDI files in {midi_dir}")

    # lookup_dict = {}

    # delete all shard files in folder
    for f in out_dir.glob("shard_*.bin"):
        f.unlink()

    skipped = 0
    tokens_so_far = 0
    for i, f in enumerate(files):
        try:
            tokens = midi_to_tokens(str(f))
            # from nanochat.tokenizer_pre_update import tokens_to_midi
            # tokens_to_midi(tokens, 'tmp.mid')

            # save as "shard"
            path = out_dir / f"shard_{i:04d}_{len(tokens):04d}.bin"
            np.array(tokens, dtype=np.uint16).tofile(str(path))
            # print(tokens)
            tokens_so_far += len(tokens)
            # print(np.fromfile(path, dtype=np.uint16).astype(np.int64))

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  skip {f.name}: {e}")
        if (i + 1) % 200 == 0:
            # print(f"  {i+1}/{len(files)}  tokens so far: {len(all_tokens):,}")
            print(f"  {i+1}/{len(files)}  tokens so far: {tokens_so_far:,}")

        # if i > 800:
        #     break


    
    # with open(os.path.join(DEFAULT_CACHE,'note_lookup.pkl'), 'wb') as f:
    #     pickle.dump(lookup_dict, f)

    # print(f"Total: {len(all_tokens):,} tokens  (skipped {skipped})")
    # arr = np.array(all_tokens, dtype=np.uint16)
    # n_shards = max(1, len(arr) // shard_size)
    # for idx, shard in enumerate(np.array_split(arr, n_shards)):
    #     path = out_dir / f"shard_{idx:04d}.bin"
    #     shard.tofile(str(path))
    #     print(f"  wrote {path}  ({len(shard):,} tokens)")
    # print(f"Done — {n_shards} shard(s) in {out_dir}")

    print(f"Total: {tokens_so_far:,} tokens  (skipped {skipped})")
    print(f"Done — {len(files)} shard(s) in {out_dir}")

if __name__ == "__main__":
# if True:
    p = argparse.ArgumentParser()
    # p.add_argument("--midi-dir",   required=True)
    p.add_argument("--midi-dir",   default=os.path.expanduser("~/.cache/nanochat/base_data_midi"))
    p.add_argument("--out-dir",    default=DATA_DIR)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = p.parse_args()
    tokenize_directory(args.midi_dir, args.out_dir, args.shard_size)

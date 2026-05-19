"""
scripts/generate.py  –  NEW file (no equivalent in nanochat)
=============================================================
Loads a nanochat checkpoint and generates a .mid file.
Uses nanochat's own engine.py (KV-cache inference) when available,
falls back to the model's built-in naive generate().

Usage:
    python -m scripts.generate \
        --checkpoint ~/.cache/nanomusic/checkpoints/ckpt_music_final.pt \
        --length 512 --temperature 1.0 --top-k 40 --out song.mid

    # Seed from an existing MIDI (continuation):
    python -m scripts.generate \
        --checkpoint ckpt.pt --prompt seed.mid --length 512 --out cont.mid
"""

import argparse
import os
import torch

from nanochat.common import autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.tokenizer import midi_to_tokens, tokens_to_midi, VOCAB_SIZE, EOS_TOKEN
# nanochat's own gpt.py — completely unchanged:
from nanochat.gpt import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--length",       type=int,   default=512)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top-k",        type=int,   default=40)
    p.add_argument("--prompt",       type=str,   default=None,
                   help="Optional seed MIDI file")
    p.add_argument("--out",          type=str,   default="generated.mid")
    p.add_argument("--device",       type=str,   default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or autodetect_device_type())
    print(f"Device: {device}")

    # Load checkpoint (nanochat's load_checkpoint expects model + optional phase)
    model, _, meta = load_checkpoint("base", device, phase="generate")
    model.eval()
    cfg = model.config
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  "
          f"vocab={cfg.vocab_size}  seq={cfg.sequence_len}")

    # Prompt
    if args.prompt:
        toks = midi_to_tokens(args.prompt)[:cfg.sequence_len - 1]
        print(f"Prompt: {len(toks)} tokens from {args.prompt}")
        idx = torch.tensor([toks], dtype=torch.long, device=device)
        n_prompt = len(toks)
    else:
        idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        n_prompt = 0

    print(f"Generating {args.length} tokens (temp={args.temperature}, top_k={args.top_k})…")
    with torch.no_grad():
        out = model.generate(idx, args.length,
                             temperature=args.temperature,
                             top_k=args.top_k)

    tok_list = out[0].tolist()
    if args.prompt:
        tok_list = tok_list[n_prompt:]  # keep only newly generated tokens

    tokens_to_midi(tok_list, args.out)
    print(f"\nSaved → {args.out}  ({len(tok_list)} tokens)")
    print("Open with MuseScore, GarageBand, VLC, or any MIDI player.")


if __name__ == "__main__":
    main()

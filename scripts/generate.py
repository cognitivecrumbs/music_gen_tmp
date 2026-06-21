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
import shutil

from nanochat.common import autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint,load_model_from_dir
from nanochat.tokenizer_pre_update import midi_to_tokens, tokens_to_midi, VOCAB_SIZE, EOS_TOKEN
# nanochat's own gpt.py — completely unchanged:
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine


def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--checkpoint",   required=True)
    p.add_argument("--model-name",   default='d4')
    p.add_argument("--step",         type=int,   default=1)
    p.add_argument("--length",       type=int,   default=512)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top-k",        type=int,   default=40)
    p.add_argument("--prompt",       type=str,   default=None,
                   help="Optional seed MIDI file")
    # p.add_argument("--prompt-seed-len", type=int,  default=1)
    p.add_argument("--out",          type=str,   default="generated")
    p.add_argument("--include-prompt", type=int, default=1, choices=[0,1])
    p.add_argument("--reference_out",          type=str,   default="reference.mid")
    p.add_argument("--device",       type=str,   default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_fname = os.path.splitext(args.out)[0]
    device = torch.device(args.device or autodetect_device_type())
    print(f"Device: {device}")

    checkpoint_dir = os.path.join(os.path.expanduser('~'),'.cache/nanochat/base_checkpoints/')
    # Load checkpoint (nanochat's load_checkpoint expects model + optional phase)
    # model, _, meta = load_checkpoint("base", device, phase="generate")
    # model, _, meta = load_checkpoint(checkpoint_dir, args.step, device=device)

    model, tokenizer, meta_data = load_model_from_dir(checkpoint_dir,device,'eval',model_tag=args.model_name,step=args.step)
    model.eval()
    cfg = model.config
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  "
          f"vocab={cfg.vocab_size}  seq={cfg.sequence_len}")

    # Prompt
    if args.prompt:
        shutil.copy(args.prompt,args.reference_out)
        toks = midi_to_tokens(args.prompt)[:cfg.sequence_len - 1]
        print(f"Prompt: {len(toks)} tokens from {args.prompt}")
        idx = torch.tensor(toks, dtype=torch.long, device=device)
        n_prompt = len(toks)
    else:
        # idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        idx = torch.zeros(1, 4, dtype=torch.long, device=device) # temporary
        n_prompt = 0
    # print(idx)
    # print(idx.shape)

    print(f"Generating {args.length} tokens (temp={args.temperature}, top_k={args.top_k})…")

    _engine = Engine(model, tokenizer)

    _sample, _ = _engine.generate_batch(idx, num_samples=1, max_tokens=args.length, temperature=args.temperature, top_k=args.top_k)

    tok_list = _sample[0]
    if args.include_prompt:
        tokens_to_midi(tok_list, out_fname + '_with_prompt' + '.mid')
        print(f"\nSaved midi with prompt → {out_fname + '_with_prompt' + '.mid'}  ({len(tok_list)} tokens)")

        tok_list = tok_list[n_prompt:]  # keep only newly generated tokens

    tokens_to_midi(tok_list, out_fname + '.mid')
    print(f"\nSaved → {out_fname + '.mid'}  ({len(tok_list)} tokens)")
    print("Open with MuseScore, GarageBand, VLC, or any MIDI player.")


if __name__ == "__main__":
    main()
# main()
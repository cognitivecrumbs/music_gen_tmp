"""

flat remi like tokenizer
nanochat/tokenizer.py  –  REPLACE the original
===============================================
Must satisfy every call site in nanochat's codebase:

base_train.py:
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)        # <-- device kwarg
    vocab_size = tokenizer.get_vocab_size()

dataloader.py:
    bos_token = tokenizer.get_bos_token_id()
    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=N)

loss_eval.py / core_eval.py:
    from nanochat.tokenizer import HuggingFaceTokenizer  # must be importable

engine.py:
    tokens = tokenizer(prompt, prepend="<|bos|>")       # __call__
    tokenizer.decode(sample[0])
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import List, Optional, Union

# ---- MIDI vocabulary -----------------------------------------------------
NOTE_ON_OFFSET    = 0
NOTE_OFF_OFFSET   = 128
TIME_SHIFT_OFFSET = 256
VELOCITY_OFFSET   = 356
PEDAL_ON          = 388
PEDAL_OFF         = 389
BAR_TOKEN         = 390
EOS_TOKEN         = 391
VOCAB_SIZE        = 512

TIME_SHIFT_STEPS  = 100
TIME_SHIFT_MS     = 10
VELOCITY_BINS     = 32


# ---- encode/decode helpers -----------------------------------------------

def velocity_to_bin(v: int) -> int:
    return min(VELOCITY_BINS - 1, v * VELOCITY_BINS // 128)

def bin_to_velocity(b: int) -> int:
    return b * 128 // VELOCITY_BINS + (128 // VELOCITY_BINS) // 2

def time_ms_to_tokens(ms: float) -> List[int]:
    ms = max(0, int(round(ms)))
    out = []
    while ms > 0:
        steps = min(TIME_SHIFT_STEPS, (ms + TIME_SHIFT_MS - 1) // TIME_SHIFT_MS)
        steps = max(1, steps)
        out.append(TIME_SHIFT_OFFSET + steps - 1)
        ms -= steps * TIME_SHIFT_MS
    return out or [TIME_SHIFT_OFFSET]

def token_to_time_ms(token: int) -> float:
    return (token - TIME_SHIFT_OFFSET + 1) * TIME_SHIFT_MS


# ---- MIDI file parser (pure stdlib) ---------------------------------------

def _read_varlen(data: bytes, pos: int):
    value = 0
    while True:
        byte = data[pos]; pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos

def midi_to_tokens(midi_path: str) -> List[int]:
    data = Path(midi_path).read_bytes()
    pos = 0
    assert data[pos:pos+4] == b'MThd', "Not a MIDI file"
    pos += 4
    hdr_len = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
    pos += 2  # format
    n_tracks = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    ticks_per_beat = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    pos += hdr_len - 6

    tempo = 500000
    ms_per_tick = tempo / (ticks_per_beat * 1000.0)
    all_events = []

    for _ in range(n_tracks):
        while pos < len(data) and data[pos:pos+4] != b'MTrk':
            pos += 1
        if pos >= len(data): break
        pos += 4
        track_len = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
        track_end = pos + track_len
        abs_tick = 0; rs = 0
        while pos < track_end:
            delta, pos = _read_varlen(data, pos)
            abs_tick += delta
            if pos >= track_end: break
            byte = data[pos]
            if byte == 0xFF:
                pos += 1; mt = data[pos]; pos += 1
                ml, pos = _read_varlen(data, pos)
                if mt == 0x51 and ml == 3:
                    tempo = struct.unpack('>I', b'\x00' + data[pos:pos+3])[0]
                    ms_per_tick = tempo / (ticks_per_beat * 1000.0)
                pos += ml; continue
            if byte in (0xF0, 0xF7):
                pos += 1; sl, pos = _read_varlen(data, pos); pos += sl; continue
            if byte & 0x80: rs = byte; pos += 1
            mt2 = rs & 0xF0
            if mt2 in (0x80, 0x90):
                pitch = data[pos]; pos += 1; vel = data[pos]; pos += 1
                on = (mt2 == 0x90) and (vel > 0)
                all_events.append((abs_tick, 'on' if on else 'off', pitch, vel))
            elif mt2 == 0xB0:
                ctrl = data[pos]; pos += 1; val = data[pos]; pos += 1
                if ctrl == 64:
                    all_events.append((abs_tick, 'pedal', val >= 64, 0))
            elif mt2 in (0xA0, 0xC0, 0xD0, 0xE0):
                pos += 1 if mt2 in (0xC0, 0xD0) else 2
            else:
                pos += 1
        pos = track_end

    if not all_events:
        return [EOS_TOKEN]

    all_events.sort(key=lambda e: e[0])
    tokens = []
    prev_tick = 0
    prev_vbin = velocity_to_bin(64)

    for ev in all_events:
        tick = ev[0]
        dt_ms = (tick - prev_tick) * ms_per_tick
        prev_tick = tick
        tokens.extend(time_ms_to_tokens(dt_ms))
        if ev[1] == 'on':
            vb = velocity_to_bin(ev[3])
            if vb != prev_vbin:
                tokens.append(VELOCITY_OFFSET + vb); prev_vbin = vb
            tokens.append(NOTE_ON_OFFSET + ev[2])
        elif ev[1] == 'off':
            tokens.append(NOTE_OFF_OFFSET + ev[2])
        elif ev[1] == 'pedal':
            tokens.append(PEDAL_ON if ev[2] else PEDAL_OFF)

    tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_midi(tokens: List[int], output_path: str,
                   ticks_per_beat: int = 480, tempo: int = 500000) -> str:
    ms_per_tick = tempo / (ticks_per_beat * 1000.0)
    ticks_per_ms = 1.0 / ms_per_tick
    events = []; abs_ms = 0.0; velocity = 64

    for tok in tokens:
        if tok == EOS_TOKEN: break
        if tok == BAR_TOKEN: continue
        elif NOTE_ON_OFFSET <= tok < NOTE_ON_OFFSET + 128:
            events.append((int(abs_ms * ticks_per_ms), 0x90, tok - NOTE_ON_OFFSET, velocity))
        elif NOTE_OFF_OFFSET <= tok < NOTE_OFF_OFFSET + 128:
            events.append((int(abs_ms * ticks_per_ms), 0x80, tok - NOTE_OFF_OFFSET, 0))
        elif TIME_SHIFT_OFFSET <= tok < TIME_SHIFT_OFFSET + TIME_SHIFT_STEPS:
            abs_ms += token_to_time_ms(tok)
        elif VELOCITY_OFFSET <= tok < VELOCITY_OFFSET + VELOCITY_BINS:
            velocity = bin_to_velocity(tok - VELOCITY_OFFSET)
        elif tok == PEDAL_ON:
            events.append((int(abs_ms * ticks_per_ms), 0xB0, 64, 127))
        elif tok == PEDAL_OFF:
            events.append((int(abs_ms * ticks_per_ms), 0xB0, 64, 0))

    events.sort(key=lambda e: e[0])

    def varlen(v):
        buf = [v & 0x7F]; v >>= 7
        while v: buf.append((v & 0x7F) | 0x80); v >>= 7
        return bytes(reversed(buf))

    track = bytearray()
    prev = 0
    for ev in events:
        track.extend(varlen(ev[0] - prev)); prev = ev[0]
        track.extend(bytes(ev[1:]))
    track.extend(b'\x00\xFF\x2F\x00')

    tempo_ev = b'\x00\xFF\x51\x03' + struct.pack('>I', tempo)[1:]
    full_track = bytearray(tempo_ev) + track
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('>4sIHHH', b'MThd', 6, 0, 1, ticks_per_beat))
        f.write(struct.pack('>4sI', b'MTrk', len(full_track)))
        f.write(full_track)
    return output_path


# ---- nanochat tokenizer interface ----------------------------------------

class MidiTokenizer:
    """
    Satisfies every call nanochat makes on the tokenizer object.
    """

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE

    def get_bos_token_id(self) -> int:
        # MIDI has no real BOS; reuse EOS as a neutral padding/start token
        return EOS_TOKEN

    def encode(self,
               texts_or_paths,
               prepend: Optional[Union[int, str]] = None,
               num_threads: int = 1) -> List[List[int]]:
        """
        nanochat calls this with a list of text strings from parquet rows.
        Our dataloader (02b_dataloader.py) never calls encode() on real strings —
        tokens are pre-computed. This shim handles the call gracefully anyway.
        """
        bos = self.get_bos_token_id() if prepend == "<|bos|>" else (prepend if isinstance(prepend, int) else None)
        out = []
        for item in (texts_or_paths or []):
            try:
                toks = midi_to_tokens(str(item))
            except Exception:
                toks = [EOS_TOKEN]
            if bos is not None:
                toks = [bos] + toks
            out.append(toks)
        return out

    def decode(self, tokens) -> str:
        """Not meaningful for MIDI but satisfies the interface."""
        return f"<midi:{len(tokens)}_tokens>"

    def __call__(self, text_or_path, prepend=None):
        """engine.py calls tokenizer(prompt, prepend='<|bos|>')"""
        result = self.encode([text_or_path], prepend=prepend)
        return result[0] if result else [EOS_TOKEN]

    def encode_special(self, token_str: str) -> int:
        """
        Engine.generate_batch() calls this to get the assistant_end stop token.
        For MIDI there is no special stop token — return EOS_TOKEN so generation
        stops naturally when the model emits EOS, and never on assistant_end.
        Any unrecognised special string also returns EOS_TOKEN.
        """
        return EOS_TOKEN

    def __len__(self):
        return VOCAB_SIZE


# HuggingFaceTokenizer alias — imported by loss_eval.py and core_eval.py.
# Those modules aren't used in nanomusic training, but the import must not crash.
HuggingFaceTokenizer = MidiTokenizer


def get_tokenizer(base_dir=None) -> MidiTokenizer:
    """nanochat API: returns the tokenizer."""
    return MidiTokenizer()


def get_token_bytes(tokenizer=None, device=None) -> dict:
    """
    nanochat calls: get_token_bytes(device=device)
    Used by loss_eval.py to compute bits-per-byte.
    For MIDI there's no byte mapping; return 1.0 per token so BPB = loss/ln(2).
    The 'device' and 'tokenizer' kwargs are accepted and ignored.
    """
    return {i: 1.0 for i in range(VOCAB_SIZE)}
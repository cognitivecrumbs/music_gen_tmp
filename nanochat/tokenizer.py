"""
nanochat/tokenizer.py  —  OctupleMIDI tokenizer (MusicBERT / Zeng et al. 2021)
===============================================================================
Each note is emitted as exactly 8 consecutive tokens, one per field:
    (time_sig, tempo, bar, position, instrument, pitch, duration, velocity)

Token IDs are drawn from non-overlapping ranges so OctupleEmbedding in gpt.py
can identify each field by ID alone.

Vocabulary (matches OCTOPLE_FIELD_VALS = [256]*8 in gpt.py):
    [   0 ..  255]  TIME_SIG
    [ 256 ..  511]  TEMPO
    [ 512 ..  767]  BAR
    [ 768 .. 1023]  POSITION
    [1024 .. 1279]  INSTRUMENT
    [1280 .. 1535]  PITCH
    [1536 .. 1791]  DURATION
    [1792 .. 2047]  VELOCITY
    [2048]          EOS   ← single int, repeated 8 times to pad to note boundary
    [2049]          PAD

VOCAB_SIZE = 2050

Inference note
--------------
engine.py generates ONE token at a time. For OctupleMIDI we need 8 tokens
per note. The solution: engine.py already has a `forced_tokens` deque per row.
After generating the first field of a note (TIME_SIG), we force-inject the
remaining 7 fields. This is handled by a helper `note_fields_after_first()`
which the sampling block in base_train.py / engine.py can call.

In practice for nanomusic we skip tool-use and just generate greedily in
groups of 8 — see generate_music_batch() below which wraps Engine cleanly.
"""

from __future__ import annotations
import math
import struct
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ── Vocabulary ───────────────────────────────────────────────────────────────

FIELD_SIZE   = 256
NUM_FIELDS   = 8
BASE_VOCAB   = FIELD_SIZE * NUM_FIELDS   # 2048
BASE_VOCAB   = 256+2

F_TIME_SIG   = 0*0 * FIELD_SIZE
F_TEMPO      = 0*1 * FIELD_SIZE
F_BAR        = 0*2 * FIELD_SIZE
F_POSITION   = 0*3 * FIELD_SIZE
F_INSTRUMENT = 0*4 * FIELD_SIZE
F_PITCH      = 0*5 * FIELD_SIZE
F_DURATION   = 0*6 * FIELD_SIZE
F_VELOCITY   = 0*7 * FIELD_SIZE

FIELD_OFFSETS = [
    F_TIME_SIG, F_TEMPO, F_BAR, F_POSITION,
    F_INSTRUMENT, F_PITCH, F_DURATION, F_VELOCITY,
]

# EOS is a SINGLE integer — the engine generates one token at a time.
# We use BASE_VOCAB (2048) and pad notes to 8-token boundaries with it.
EOS_TOKEN  = BASE_VOCAB       # 2048  ← int, not a list
PAD_TOKEN  = BASE_VOCAB + 1   # 2049
VOCAB_SIZE = BASE_VOCAB + 2   # 2050
# EOS_TOKEN  = BASE_VOCAB       # 2048  ← int, not a list

EOS_TOKEN = FIELD_SIZE
PAD_TOKEN = FIELD_SIZE + 1

# 8-token EOS note: fills one complete note slot, signals end of sequence
EOS_NOTE = [EOS_TOKEN] * NUM_FIELDS


# ── Field encoders / decoders ────────────────────────────────────────────────

_TIME_SIGS = [
    (4,4),(3,4),(2,4),(6,8),(12,8),(2,2),(3,8),(5,4),
    (7,8),(6,4),(9,8),(5,8),(7,4),(11,8),(3,2),(1,4),
]
_TS_TO_IDX = {ts: i for i, ts in enumerate(_TIME_SIGS)}

def encode_time_sig(num: int, denom: int) -> int:
    return F_TIME_SIG + _TS_TO_IDX.get((num, denom), 0)

def decode_time_sig(tok: int) -> Tuple[int, int]:
    idx = tok - F_TIME_SIG
    return _TIME_SIGS[idx] if 0 <= idx < len(_TIME_SIGS) else (4, 4)

_TEMPO_MIN, _TEMPO_MAX = 32.0, 240.0

def encode_tempo(bpm: float) -> int:
    bpm = max(_TEMPO_MIN, min(_TEMPO_MAX, bpm))
    ratio = math.log(bpm / _TEMPO_MIN) / math.log(_TEMPO_MAX / _TEMPO_MIN)
    return F_TEMPO + int(ratio * (FIELD_SIZE - 1))

def decode_tempo(tok: int) -> float:
    ratio = (tok - F_TEMPO) / (FIELD_SIZE - 1)
    return _TEMPO_MIN * (_TEMPO_MAX / _TEMPO_MIN) ** ratio

def encode_bar(bar_idx: int) -> int:
    return F_BAR + (bar_idx % FIELD_SIZE)

def decode_bar(tok: int) -> int:
    return tok - F_BAR

def encode_position(pos_64ths: int) -> int:
    return F_POSITION + min(pos_64ths, FIELD_SIZE - 1)

def decode_position(tok: int) -> int:
    return tok - F_POSITION

def encode_instrument(program: int, is_drum: bool = False) -> int:
    idx = 128 if is_drum else min(program, 127)
    return F_INSTRUMENT + idx

def decode_instrument(tok: int) -> Tuple[int, bool]:
    idx = tok - F_INSTRUMENT
    return (0, True) if idx == 128 else (idx, False)

def encode_pitch(pitch: int) -> int:
    return F_PITCH + min(pitch, 128)   # 128 = rest

def decode_pitch(tok: int) -> int:
    return tok - F_PITCH

_DUR_MIN, _DUR_MAX = 10.0, 8000.0

def encode_duration(ms: float) -> int:
    ms = max(_DUR_MIN, min(_DUR_MAX, ms))
    ratio = math.log(ms / _DUR_MIN) / math.log(_DUR_MAX / _DUR_MIN)
    return F_DURATION + int(ratio * (FIELD_SIZE - 1))

def decode_duration(tok: int) -> float:
    ratio = (tok - F_DURATION) / (FIELD_SIZE - 1)
    return _DUR_MIN * (_DUR_MAX / _DUR_MIN) ** ratio

def encode_velocity(vel: int) -> int:
    return F_VELOCITY + min(vel, FIELD_SIZE - 1)

def decode_velocity(tok: int) -> int:
    return tok - F_VELOCITY


# ── MIDI parser ───────────────────────────────────────────────────────────────

def _read_varlen(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]; pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos


def _parse_midi(path: str):
    data = Path(path).read_bytes()
    pos = 0
    assert data[pos:pos+4] == b'MThd'
    pos += 4
    hdr_len        = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
    pos += 2
    n_tracks       = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    ticks_per_beat = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    pos += hdr_len - 6

    all_events = []

    for _ in range(n_tracks):
        while pos < len(data) and data[pos:pos+4] != b'MTrk':
            pos += 1
        if pos >= len(data): break
        pos += 4
        track_len = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
        track_end = pos + track_len
        abs_tick = 0; rs = 0; program = [0] * 16

        while pos < track_end:
            delta, pos = _read_varlen(data, pos)
            abs_tick  += delta
            if pos >= track_end: break
            byte = data[pos]
            if byte == 0xFF:
                pos += 1; mt = data[pos]; pos += 1
                ml, pos = _read_varlen(data, pos)
                if mt == 0x51 and ml == 3:
                    t = struct.unpack('>I', b'\x00' + data[pos:pos+3])[0]
                    all_events.append((abs_tick, 'tempo', 60_000_000 / t if t > 0 else 120.0))
                elif mt == 0x58 and ml >= 2:
                    all_events.append((abs_tick, 'timesig', data[pos], data[pos+1]))
                pos += ml; continue
            if byte in (0xF0, 0xF7):
                pos += 1; sl, pos = _read_varlen(data, pos); pos += sl; continue
            if byte & 0x80: rs = byte; pos += 1
            ch = rs & 0x0F; mt2 = rs & 0xF0
            if mt2 == 0xC0:
                program[ch] = data[pos]; pos += 1
            elif mt2 in (0x80, 0x90):
                pitch = data[pos]; pos += 1; vel = data[pos]; pos += 1
                on = (mt2 == 0x90 and vel > 0)
                all_events.append((abs_tick, 'on' if on else 'off',
                                   pitch, vel, program[ch], ch == 9, ch))
            elif mt2 in (0xA0, 0xB0, 0xE0): pos += 2
            elif mt2 in (0xC0, 0xD0):        pos += 1
            else:                             pos += 1
        pos = track_end

    all_events.sort(key=lambda e: e[0])

    open_notes = {}
    notes = []
    tempos    = [(0, 120.0)]
    time_sigs = [(0, 4, 4)]

    for ev in all_events:
        if ev[1] == 'tempo':
            tempos.append((ev[0], ev[2]))
        elif ev[1] == 'timesig':
            time_sigs.append((ev[0], ev[2], 2 ** ev[3]))
        elif ev[1] == 'on':
            _, _, pitch, vel, prog, is_drum, ch = ev
            open_notes[(pitch, ch)] = (ev[0], vel, prog, is_drum)
        elif ev[1] == 'off':
            _, _, pitch, vel, prog, is_drum, ch = ev
            key = (pitch, ch)
            if key in open_notes:
                start, v2, p2, d2 = open_notes.pop(key)
                dur = ev[0] - start
                if dur > 0:
                    notes.append((start, pitch, v2, dur, p2, d2))

    return ticks_per_beat, notes, tempos, time_sigs


def midi_to_tokens(midi_path: str) -> List[int]:
    """MIDI file → flat OctupleMIDI token stream. Ends with EOS_NOTE (8 EOS tokens)."""
    try:
        ticks_per_beat, notes, tempos, time_sigs = _parse_midi(midi_path)
    except Exception:
        return list(EOS_NOTE)
    if not notes:
        return list(EOS_NOTE)

    def bpm_at(tick):
        bpm = 120.0
        for t, b in tempos:
            if t <= tick: bpm = b
            else: break
        return bpm

    def ts_at(tick):
        num, den = 4, 4
        for t, n, d in time_sigs:
            if t <= tick: num, den = n, d
            else: break
        return num, den

    notes.sort(key=lambda n: n[0])
    tokens: List[int] = []

    for (abs_tick, pitch, vel, dur_ticks, prog, is_drum) in notes:
        bpm      = bpm_at(abs_tick)
        num, den = ts_at(abs_tick)

        ticks_per_bar  = int(ticks_per_beat * 4 * num / den)
        bar_idx        = abs_tick // ticks_per_bar if ticks_per_bar > 0 else 0
        pos_tick       = abs_tick  % ticks_per_bar if ticks_per_bar > 0 else 0
        ticks_per_64th = max(1, ticks_per_beat // 16)
        pos_64ths      = pos_tick // ticks_per_64th

        us_per_tick = (60_000_000 / bpm) / ticks_per_beat
        dur_ms      = dur_ticks * us_per_tick / 1000.0

        tokens += [
            encode_time_sig(num, den),
            encode_tempo(bpm),
            encode_bar(bar_idx),
            encode_position(pos_64ths),
            encode_instrument(prog, is_drum),
            encode_pitch(pitch),
            encode_duration(dur_ms),
            encode_velocity(vel),
        ]

    tokens += EOS_NOTE   # 8 EOS tokens = one EOS note slot
    return tokens


def tokens_to_midi(tokens: List[int], output_path: str,
                   ticks_per_beat: int = 480) -> str:
    """Flat OctupleMIDI token stream → MIDI file."""
    # Remove trailing EOS/PAD tokens, then pad to multiple of 8
    clean = [t for t in tokens if t != PAD_TOKEN]
    while len(clean) % NUM_FIELDS:
        clean.append(PAD_TOKEN)

    events = []
    ch_map: dict = {}

    def get_ch(prog, is_drum):
        if is_drum: return 9
        if prog not in ch_map:
            used = set(ch_map.values()) | {9}
            # ch_map[prog] = next(c for c in range(16) if c not in used)
            ch_map[prog] = next(c for c in range(256) if c not in used)
        return ch_map[prog]

    for i in range(0, len(clean), NUM_FIELDS):
        tup = clean[i:i+NUM_FIELDS]
        # Skip any note whose first token is EOS or PAD
        if tup[0] >= BASE_VOCAB:
            continue

        ts_tok, tempo_tok, bar_tok, pos_tok, inst_tok, pitch_tok, dur_tok, vel_tok = tup

        num, den      = decode_time_sig(ts_tok)
        bpm           = decode_tempo(tempo_tok)
        bar_idx       = decode_bar(bar_tok)
        pos_64ths     = decode_position(pos_tok)
        prog, is_drum = decode_instrument(inst_tok)
        pitch         = decode_pitch(pitch_tok)
        dur_ms        = decode_duration(dur_tok)
        vel           = decode_velocity(vel_tok)

        if pitch >= 128: continue   # rest
        vel = max(1, min(127, vel))

        tempo_us       = int(60_000_000 / max(bpm, 1.0))
        us_per_tick    = tempo_us / ticks_per_beat
        ticks_per_bar  = int(ticks_per_beat * 4 * num / den)
        ticks_per_64th = max(1, ticks_per_beat // 16)
        abs_tick       = bar_idx * ticks_per_bar + pos_64ths * ticks_per_64th
        dur_ticks      = max(1, int(dur_ms * 1000 / us_per_tick))
        ch             = get_ch(prog, is_drum)

        if not is_drum and prog not in ch_map:
            events.append((abs_tick, bytes([0xC0 | ch, prog])))
        events.append((abs_tick,              bytes([0x90 | ch, pitch, vel])))
        events.append((abs_tick + dur_ticks,  bytes([0x80 | ch, pitch, 0])))

    events.sort(key=lambda e: e[0])

    def varlen(v):
        buf = [v & 0x7F]; v >>= 7
        while v: buf.append((v & 0x7F) | 0x80); v >>= 7
        return bytes(reversed(buf))

    track = bytearray(b'\x00\xFF\x51\x03' + struct.pack('>I', 500000)[1:])
    prev = 0
    for tick, msg in events:
        track += varlen(max(0, tick - prev)) + msg; prev = tick
    track += b'\x00\xFF\x2F\x00'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('>4sIHHH', b'MThd', 6, 0, 1, ticks_per_beat))
        f.write(struct.pack('>4sI',    b'MTrk', len(track)))
        f.write(track)
    return output_path


# ── Music generation helper ───────────────────────────────────────────────────

def generate_music(engine, tokenizer, max_notes: int = 128,
                   temperature: float = 1.0, top_k: int = 40) -> List[int]:
    """
    Generate a sequence of OctupleMIDI tokens using the Engine.

    Because engine.generate() yields ONE token at a time and OctupleMIDI
    needs exactly 8 tokens per note, we use the forced_tokens deque:
    - Sample the TIME_SIG token (field 0) for each note position
    - Force-inject fields 1-7 from the model's subsequent predictions
    - Collect all 8 tokens per note into a flat list
    - Stop when EOS_TOKEN is sampled or max_notes reached

    This requires no changes to engine.py — we just collect the token
    stream and group it into notes of 8 afterward.
    """
    # prompt = [EOS_TOKEN]   # single EOS as prompt (neutral start)
    prompt = list(EOS_NOTE)   # single EOS as prompt (neutral start)
    max_tokens = max_notes * NUM_FIELDS

    all_tokens: List[int] = []
    for token_column, _ in engine.generate(
            prompt, num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k):
        tok = token_column[0]
        all_tokens.append(tok)
        # Stop on EOS note boundary
        if tok == EOS_TOKEN and len(all_tokens) % NUM_FIELDS == 0:
            break

    return all_tokens


# ── nanochat tokenizer interface ──────────────────────────────────────────────

class MidiTokenizer:

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE

    def get_bos_token_id(self) -> int:
        # Return a single int — engine.py passes this to the prompt list.
        # We use EOS_TOKEN as a neutral start-of-sequence marker.
        # return EOS_TOKEN
        return EOS_NOTE

    def encode_special(self, token_str: str) -> int:
        # engine.py calls this for "<|assistant_end|>", "<|python_start|>", etc.
        # All unknown special strings → EOS_TOKEN so generation terminates cleanly.
        # return EOS_TOKEN
        return EOS_NOTE

    def encode(self, items, prepend=None, num_threads: int = 1) -> List[List[int]]:
        bos = (self.get_bos_token_id()
               if prepend in ("<|bos|>", EOS_TOKEN)
               else (prepend if isinstance(prepend, int) else None))
        out = []
        for item in (items or []):
            try:    toks = midi_to_tokens(str(item))
            except: toks = list(EOS_NOTE)
            if bos is not None:
                toks = [bos] + toks
            out.append(toks)
        return out

    def decode(self, tokens) -> str:
        return f"<midi:{len(tokens)}_tokens>"

    def __call__(self, item, prepend=None):
        return self.encode([item], prepend=prepend)[0]

    def __len__(self):
        return VOCAB_SIZE


HuggingFaceTokenizer = MidiTokenizer

def get_tokenizer(base_dir=None) -> MidiTokenizer:
    return MidiTokenizer()

def get_token_bytes(tokenizer=None, device=None) -> dict:
    return {i: 1.0 for i in range(VOCAB_SIZE)}
"""
nanochat/tokenizer.py  –  OctupleMIDI-style tokenizer
======================================================
Replaces the flat event-based tokenizer with OctupleMIDI encoding
(MusicBERT, Zeng et al. 2021).

OctupleMIDI encodes each note as a fixed tuple of 8 tokens emitted
in sequence:
  (time_sig, tempo, bar, position, instrument, pitch, duration, velocity)

This gives ~4x shorter sequences than REMI and makes the structure
explicit so the model can learn bar/position/tempo relationships.

Vocabulary layout (2048 tokens total, 256 per field):
  [0*256 .. 1*256)   TIME_SIG   — 256 buckets (num/denom combos)
  [1*256 .. 2*256)   TEMPO      — 256 buckets (32–240 BPM log-scaled)
  [2*256 .. 3*256)   BAR        — 256 bar indices (wraps at 256)
  [3*256 .. 4*256)   POSITION   — 256 positions within a bar (1/64 note grid)
  [4*256 .. 5*256)   INSTRUMENT — 256 (GM program 0-127 + 128 = percussion)
  [5*256 .. 6*256)   PITCH      — 256 (MIDI pitch 0-127, 128 = rest)
  [6*256 .. 7*256)   DURATION   — 256 buckets (log-scaled 10ms–8s)
  [7*256 .. 8*256)   VELOCITY   — 256 (MIDI velocity 0-127 + 0 = silent)

Special tokens (appended after the 8×256 = 2048 block):
  2048  EOS
  2049  PAD

VOCAB_SIZE = 2050

nanochat interface satisfied:
  get_tokenizer()              → MidiTokenizer
  get_token_bytes(device=)     → dict
  tokenizer.get_vocab_size()
  tokenizer.get_bos_token_id()
  tokenizer.encode_special(s)  → int
  tokenizer.encode(batch, prepend=, num_threads=)
  tokenizer.decode(tokens)     → str
  tokenizer(prompt, prepend=)  → list[int]
  HuggingFaceTokenizer         alias
"""

from __future__ import annotations
import math
import struct
import pickle
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple

from nanochat.tokenizer import tokens_to_midi

# ── Vocabulary layout ────────────────────────────────────────────────────────

DEFAULT_CACHE = os.path.expanduser("~/.cache/nanochat")

TIME_SIG_ENCODE   = False
TEMPO_ENCODE      = False
BAR_ENCODE        = True
POSITION_ENCODE   = True
INSTRUMENT_ENCODE = False
PITCH_ENCODE      = True
DURATION_ENCODE   = True
VELOCITY_ENCODE   = False

FIELD_SIZE   = 256          # tokens per field
# NUM_FIELDS   = 8
NUM_FIELDS   = sum([
    TIME_SIG_ENCODE, TEMPO_ENCODE, BAR_ENCODE, POSITION_ENCODE,INSTRUMENT_ENCODE,PITCH_ENCODE,DURATION_ENCODE,VELOCITY_ENCODE])
BASE_VOCAB   = FIELD_SIZE * NUM_FIELDS   # 2048


# Field offsets
F_TIME_SIG   = 0 * FIELD_SIZE
F_TEMPO      = 1 * FIELD_SIZE
F_BAR        = 2 * FIELD_SIZE
F_POSITION   = 3 * FIELD_SIZE
F_INSTRUMENT = 4 * FIELD_SIZE
F_PITCH      = 5 * FIELD_SIZE
F_DURATION   = 6 * FIELD_SIZE
F_VELOCITY   = 7 * FIELD_SIZE

F_TIME_SIG   = 0
F_TEMPO      = 0
F_BAR        = 0
F_POSITION   = 0
F_INSTRUMENT = 0
F_PITCH      = 0
F_DURATION   = 0
F_VELOCITY   = 0

# F_TIME_SIG   = 0
# if TIME_SIG_ENCODE:
#     F_TEMPO      = FIELD_SIZE
# F_BAR = F_TIME_SIG
# if TEMPO_ENCODE:
#     F_BAR        = F_TEMPO + FIELD_SIZE
# F_POSITION = F_BAR
# if BAR_ENCODE:
#     F_POSITION   = F_BAR + FIELD_SIZE
# F_INSTRUMENT = F_POSITION
# if POSITION_ENCODE:
#     F_INSTRUMENT = F_POSITION + FIELD_SIZE
# F_PITCH = F_INSTRUMENT
# if INSTRUMENT_ENCODE:
#     F_PITCH      = F_INSTRUMENT + FIELD_SIZE
# F_DURATION = F_PITCH
# if PITCH_ENCODE:
#     F_DURATION   = F_PITCH + FIELD_SIZE
# F_VELOCITY = F_DURATION
# if DURATION_ENCODE:
#     F_VELOCITY   = F_DURATION + FIELD_SIZE



EOS_TOKEN    = BASE_VOCAB        # 2048
# EOS_TOKEN    = 0        # 2048
PAD_TOKEN    = BASE_VOCAB + 1    # 2049
VOCAB_SIZE   = BASE_VOCAB + 2    # 2050

EOS_TOKEN    = [[255]*NUM_FIELDS]
PAD_TOKEN    = EOS_TOKEN
VOCAB_SIZE   = FIELD_SIZE + 1

# ── Field encoding helpers ───────────────────────────────────────────────────

# TIME_SIG: encode (numerator, denominator) pairs
# Common time sigs mapped to 0-based index; unknown → 0
_TIME_SIGS = [
    (4,4),(3,4),(2,4),(6,8),(12,8),(2,2),(3,8),(5,4),(7,8),(6,4),
    (9,8),(5,8),(7,4),(11,8),(3,2),(1,4),
]
_TS_TO_IDX = {ts: i for i, ts in enumerate(_TIME_SIGS)}

def encode_time_sig(num: int, denom: int) -> int:
    idx = _TS_TO_IDX.get((num, denom), 0)
    return F_TIME_SIG + min(idx, FIELD_SIZE - 1)

def decode_time_sig(tok: int) -> Tuple[int, int]:
    idx = tok - F_TIME_SIG
    if 0 <= idx < len(_TIME_SIGS):
        return _TIME_SIGS[idx]
    return (4, 4)

# TEMPO: log-scale 32–240 BPM → 256 buckets
_TEMPO_MIN, _TEMPO_MAX = 32.0, 240.0

def encode_tempo(bpm: float) -> int:
    bpm   = max(_TEMPO_MIN, min(_TEMPO_MAX, bpm))
    ratio = math.log(bpm / _TEMPO_MIN) / math.log(_TEMPO_MAX / _TEMPO_MIN)
    idx   = int(ratio * (FIELD_SIZE - 1))
    return F_TEMPO + idx

def decode_tempo(tok: int) -> float:
    idx = tok - F_TEMPO
    ratio = idx / (FIELD_SIZE - 1)
    return _TEMPO_MIN * (_TEMPO_MAX / _TEMPO_MIN) ** ratio

# BAR: bar index mod 256
def encode_bar(bar_idx: int) -> int:
    return F_BAR + (bar_idx % FIELD_SIZE)

def decode_bar(tok: int) -> int:
    return tok - F_BAR

# POSITION: 1/64-note grid within a bar, 256 slots
def encode_position(pos_64ths: int) -> int:
    return F_POSITION + min(pos_64ths, FIELD_SIZE - 1)

def decode_position(tok: int) -> int:
    return tok - F_POSITION

# INSTRUMENT: GM program 0-127; 128 = percussion
def encode_instrument(program: int, is_drum: bool = False) -> int:
    idx = 128 if is_drum else min(program, 127)
    return F_INSTRUMENT + idx

def decode_instrument(tok: int) -> Tuple[int, bool]:
    idx = tok - F_INSTRUMENT
    if idx == 128:
        return 0, True
    return idx, False

# PITCH: MIDI pitch 0-127; 128 = rest (no note)
def encode_pitch(pitch: int) -> int:
    return F_PITCH + min(pitch, 128)

def decode_pitch(tok: int) -> int:
    return tok - F_PITCH

# DURATION: log-scale 10ms–8000ms → 256 buckets
_DUR_MIN, _DUR_MAX = 10.0, 8000.0

def encode_duration(ms: float) -> int:
    ms    = max(_DUR_MIN, min(_DUR_MAX, ms))
    ratio = math.log(ms / _DUR_MIN) / math.log(_DUR_MAX / _DUR_MIN)
    idx   = int(ratio * (FIELD_SIZE - 1))
    return F_DURATION + idx

def decode_duration(tok: int) -> float:
    idx = tok - F_DURATION
    ratio = idx / (FIELD_SIZE - 1)
    return _DUR_MIN * (_DUR_MAX / _DUR_MIN) ** ratio

# VELOCITY: MIDI 0-127 → bucket 0-127 (1:1); 0 = silent
def encode_velocity(vel: int) -> int:
    return F_VELOCITY + min(vel, FIELD_SIZE - 1)

def decode_velocity(tok: int) -> int:
    return tok - F_VELOCITY

# ── MIDI file parser (pure stdlib) ───────────────────────────────────────────

def _read_varlen(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]; pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos

def _parse_midi(path: str):
    """
    Returns (ticks_per_beat, notes, tempos, time_sigs) where:
      notes = list of (abs_tick, pitch, velocity, duration_ticks, program, is_drum)
      tempos = list of (abs_tick, bpm)
      time_sigs = list of (abs_tick, numerator, denominator)
    """
    data = Path(path).read_bytes()
    pos = 0
    assert data[pos:pos+4] == b'MThd', "Not a MIDI file"
    pos += 4
    hdr_len = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
    pos += 2  # format
    n_tracks = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    ticks_per_beat = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    pos += hdr_len - 6

    all_track_events = []  # (abs_tick, type, *data)

    for _ in range(n_tracks):
        while pos < len(data) and data[pos:pos+4] != b'MTrk':
            pos += 1
        if pos >= len(data): break
        pos += 4
        track_len = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
        track_end = pos + track_len
        abs_tick = 0; rs = 0; program = [0]*16; drum_ch = 9

        while pos < track_end:
            delta, pos = _read_varlen(data, pos)
            abs_tick += delta
            if pos >= track_end: break
            byte = data[pos]

            if byte == 0xFF:
                pos += 1; mt = data[pos]; pos += 1
                ml, pos = _read_varlen(data, pos)
                if mt == 0x51 and ml == 3:
                    t = struct.unpack('>I', b'\x00' + data[pos:pos+3])[0]
                    bpm = 60_000_000 / t if t > 0 else 120.0
                    all_track_events.append((abs_tick, 'tempo', bpm))
                elif mt == 0x58 and ml == 4:
                    num = data[pos]
                    denom = 2 ** data[pos+1]
                    all_track_events.append((abs_tick, 'time_sig', num, denom))
                pos += ml; continue

            if byte in (0xF0, 0xF7):
                pos += 1; sl, pos = _read_varlen(data, pos); pos += sl; continue

            if byte & 0x80: rs = byte; pos += 1
            ch = rs & 0x0F; mt2 = rs & 0xF0

            if mt2 == 0xC0:
                program[ch] = data[pos]; pos += 1
            elif mt2 in (0x80, 0x90):
                pitch = data[pos]; pos += 1; vel = data[pos]; pos += 1
                is_drum = (ch == drum_ch)
                prog = program[ch]
                if mt2 == 0x90 and vel > 0:
                    all_track_events.append((abs_tick, 'note_on', pitch, vel, prog, is_drum, ch))
                else:
                    all_track_events.append((abs_tick, 'note_off', pitch, ch))
            elif mt2 in (0xA0, 0xB0, 0xE0):
                pos += 2
            elif mt2 in (0xC0, 0xD0):
                pos += 1
            else:
                pos += 1
        pos = track_end

    all_track_events.sort(key=lambda e: e[0])

    # Pair note_on with note_off to get durations
    open_notes = {}  # (pitch, ch) → (abs_tick, vel, prog, is_drum)
    notes = []
    tempos = [(0, 120.0)]
    time_sigs = [(0, 4, 4)]

    for ev in all_track_events:
        if ev[1] == 'tempo':
            tempos.append((ev[0], ev[2]))
        elif ev[1] == 'time_sig':
            time_sigs.append((ev[0], ev[2], ev[3]))
        elif ev[1] == 'note_on':
            _, _, pitch, vel, prog, is_drum, ch = ev
            open_notes[(pitch, ch)] = (ev[0], vel, prog, is_drum)
        elif ev[1] == 'note_off':
            _, _, pitch, ch = ev
            key = (pitch, ch)
            if key in open_notes:
                start_tick, vel, prog, is_drum = open_notes.pop(key)
                dur_ticks = ev[0] - start_tick
                if dur_ticks > 0:
                    notes.append((start_tick, pitch, vel, dur_ticks, prog, is_drum))

    return ticks_per_beat, notes, tempos, time_sigs


# def midi_to_tokens(midi_path: str, train:bool = False) -> List[int]:
def midi_to_tokens(midi_path: str) -> List[int]:
    """
    Parse a MIDI file and return a flat list of OctupleMIDI tokens.
    Each note produces exactly 8 consecutive tokens:
      time_sig, tempo, bar, position, instrument, pitch, duration, velocity
    """
    # if len(note_lookup) == 0:
    #     note_lookup['EOS'] = EOS_TOKEN

    # tmp hardcoded sequence
    # return list(range(FIELD_SIZE)) * 10
    # return [0]*FIELD_SIZE * 10
    

    try:
        ticks_per_beat, notes, tempos, time_sigs = _parse_midi(midi_path)
    except Exception:
        # return [EOS_TOKEN]
        return EOS_TOKEN

    if not notes:
        # return [EOS_TOKEN]
        return EOS_TOKEN
 
    # if train:
    #     note_lookup = {}
    #     note_lookup['EOS'] = len(note_lookup)
    # else:
    #     note_lookup = pickle.load(open(os.path.join(DEFAULT_CACHE,'note_lookup.pkl', 'rb')))

    # Build tick→bpm lookup
    def bpm_at(tick):
        bpm = 120.0
        for t, b in tempos:
            if t <= tick: bpm = b
            else: break
        return bpm

    def time_sig_at(tick):
        num, denom = 4, 4
        for t, n, d in time_sigs:
            if t <= tick: num, denom = n, d
            else: break
        return num, denom

    notes.sort(key=lambda n: n[0])
    # tokens: List[int] = []
    tokens_list: List[List[int]] = []

    for (abs_tick, pitch, vel, dur_ticks, prog, is_drum) in notes:
        tokens: List[int] = []
        bpm      = bpm_at(abs_tick)
        num, den = time_sig_at(abs_tick)

        # Convert ticks to musical position
        ticks_per_bar = int(ticks_per_beat * 4 * num / den)
        bar_idx  = abs_tick // ticks_per_bar if ticks_per_bar > 0 else 0
        pos_tick = abs_tick % ticks_per_bar  if ticks_per_bar > 0 else 0
        # Map position to 1/64 note grid (64 positions per whole note)
        ticks_per_64th = ticks_per_beat // 16  # 1/64 = 1/4 * 1/16
        pos_64ths = pos_tick // max(1, ticks_per_64th)

        # Duration in ms
        us_per_tick = (60_000_000 / bpm) / ticks_per_beat
        dur_ms = dur_ticks * us_per_tick / 1000.0

        # tokens += [
        #     encode_time_sig(num, den),
        #     encode_tempo(bpm),
        #     encode_bar(bar_idx),
        #     encode_position(pos_64ths),
        #     encode_instrument(prog, is_drum),
        #     encode_pitch(pitch),
        #     encode_duration(dur_ms),
        #     encode_velocity(vel),
        # ]

        if TIME_SIG_ENCODE:
            tokens.append(encode_time_sig(num,den))
        if TEMPO_ENCODE:
            tokens.append(encode_tempo(bpm))
        if BAR_ENCODE:
            tokens.append(encode_bar(bar_idx))
        if POSITION_ENCODE: 
            tokens.append(encode_position(pos_64ths))
        if INSTRUMENT_ENCODE:
            tokens.append(encode_instrument(prog, is_drum))
        if PITCH_ENCODE:
            tokens.append(encode_pitch(pitch))
        if DURATION_ENCODE:
            tokens.append(encode_duration(dur_ms))
        if VELOCITY_ENCODE:
            tokens.append(encode_velocity(vel))

        tokens_list.append(tokens)



        # indiv_token = (
        #     encode_time_sig(num, den) if TIME_SIG_ENCODE else 0,
        #     encode_tempo(bpm) if TEMPO_ENCODE else 0,
        #     encode_bar(bar_idx) if BAR_ENCODE else 0,
        #     encode_position(pos_64ths) if POSITION_ENCODE else 0,
        #     encode_instrument(prog, is_drum) if INSTRUMENT_ENCODE else 0,
        #     encode_pitch(pitch) if PITCH_ENCODE else 0,
        #     encode_duration(dur_ms) if DURATION_ENCODE else 0,
        #     encode_velocity(vel) if VELOCITY_ENCODE else 0,
        # )
        # if indiv_token not in note_lookup:
        #     note_lookup[(indiv_token)] = len(note_lookup)
        # tokens += [note_lookup[indiv_token]]

    # if train:
    #     with open(os.path.join(DEFAULT_CACHE,'note_lookup.pkl'), 'wb') as f:
    #         pickle.dump(note_lookup, f)

    # tokens.append(EOS_TOKEN)
    tokens_list += [EOS_TOKEN]

    # tokens_to_midi(tokens,'out.mid')
    return tokens


def tokens_to_midi(tokens: List[int], output_path: str,
                   default_tempo: int = 500000,
                   ticks_per_beat: int = 480) -> str:
    """
    Convert OctupleMIDI tokens back to a MIDI file.
    Reads tuples of 8 tokens; ignores malformed trailing tokens.
    """
    # Strip EOS/PAD
    # clean = [t for t in tokens if t not in (EOS_TOKEN, PAD_TOKEN)]
    # clean = [tokens[i:i+NUM_FIELDS] for i in range(0, len(tokens), NUM_FIELDS) if tokens[i:i+NUM_FIELDS] not in (EOS_TOKEN, PAD_TOKEN)]
    # clean = [tokens[i:i+NUM_FIELDS] for i in range(0, len(tokens), NUM_FIELDS) if not np.any(np.all(tokens[i:i+NUM_FIELDS] == (EOS_TOKEN,PAD_TOKEN),1))]
    # clean = [tokens[i:i+NUM_FIELDS] for i in range(0, len(tokens), NUM_FIELDS) if tokens[i:i+NUM_FIELDS] != EOS_TOKEN or tokens[i:i+NUM_FIELDS] != PAD_TOKEN]
    clean = [token for token in tokens if token != EOS_TOKEN and token != PAD_TOKEN]

    # Pad to multiple of 8
    while len(clean) % NUM_FIELDS != 0:
        clean.append(PAD_TOKEN)

    events = []   # (abs_tick, msg_bytes)
    program_set = set()

    # for i in range(0, len(clean), NUM_FIELDS):
    #     tup = clean[i:i+NUM_FIELDS]
    #     if len(tup) < NUM_FIELDS:
    #         break

        # ts_tok, tempo_tok, bar_tok, pos_tok, inst_tok, pitch_tok, dur_tok, vel_tok = tup

        # # Decode fields
        # num, den       = decode_time_sig(ts_tok)
        # bpm            = decode_tempo(tempo_tok)
        # bar_idx        = decode_bar(bar_tok)
        # pos_64ths      = decode_position(pos_tok)
        # prog, is_drum  = decode_instrument(inst_tok)
        # pitch          = decode_pitch(pitch_tok)
        # dur_ms         = decode_duration(dur_tok)
        # vel            = decode_velocity(vel_tok)

        # if pitch == 128:   # rest token
        #     continue
        # vel = max(1, min(127, vel))

    for tup in clean:
        # if any(t == 256 for t in tup):
        #     continue
        if tup == [[255,255,255,255]]:
            continue
        
        # default values
        num, den, bpm, bar_idx, pos_64ths, prog, pitch, dur_ms, vel = 4, 4, 110, 0, 0, 0, 0, 0, 127
        is_drum = False
        tup_idx = 0
        if TIME_SIG_ENCODE:
            num, den = decode_time_sig(tup[tup_idx])
            tup_idx += 1
        if TEMPO_ENCODE:
            bpm = decode_tempo(tup[tup_idx])
            tup_idx += 1
        if BAR_ENCODE:
            bar_idx = decode_bar(tup[tup_idx])
            tup_idx += 1
        if POSITION_ENCODE:
            pos_64ths = decode_position(tup[tup_idx])
            tup_idx += 1
        if INSTRUMENT_ENCODE:
            prog, is_drum = decode_instrument(tup[tup_idx])
            tup_idx += 1
        if PITCH_ENCODE:
            pitch = decode_pitch(tup[tup_idx])
            # pitch = max(0, min(128, pitch))
            tup_idx += 1
        if DURATION_ENCODE:
            dur_ms = decode_duration(tup[tup_idx])
            tup_idx += 1
        if VELOCITY_ENCODE:
            vel = decode_velocity(tup[tup_idx])
            vel = max(1, min(127, vel))
            tup_idx += 1

        # Convert back to ticks
        ticks_per_bar = int(ticks_per_beat * 4 * num / den)
        ticks_per_64th = ticks_per_beat // 16
        abs_tick = bar_idx * ticks_per_bar + pos_64ths * ticks_per_64th

        us_per_beat  = 60_000_000 / bpm
        us_per_tick  = us_per_beat / ticks_per_beat
        dur_ticks    = int(dur_ms * 1000 / us_per_tick)

        ch = 9 if is_drum else min(prog % 9, 8)  # simple channel assignment
        if not is_drum and prog not in program_set:
            events.append((abs_tick, bytes([0xC0 | ch, prog])))
            program_set.add(prog)

        events.append((abs_tick,           bytes([0x90 | ch, pitch, vel])))
        events.append((abs_tick + dur_ticks, bytes([0x80 | ch, pitch, 0])))

    events.sort(key=lambda e: e[0])

    def varlen(v: int) -> bytes:
        buf = [v & 0x7F]; v >>= 7
        while v: buf.append((v & 0x7F) | 0x80); v >>= 7
        return bytes(reversed(buf))

    track = bytearray()
    # Tempo meta at start
    us = int(60_000_000 / 120)
    track += b'\x00\xFF\x51\x03' + struct.pack('>I', us)[1:]

    prev = 0
    for tick, msg in events:
        delta = max(0, tick - prev); prev = tick
        track += varlen(delta) + msg
    track += b'\x00\xFF\x2F\x00'  # end of track

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('>4sIHHH', b'MThd', 6, 0, 1, ticks_per_beat))
        f.write(struct.pack('>4sI', b'MTrk', len(track)))
        f.write(track)
    return output_path


# ── nanochat tokenizer interface ─────────────────────────────────────────────

class MidiTokenizer:
    """Satisfies every call site in nanochat's codebase."""

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE

    def get_bos_token_id(self) -> int:
        return EOS_TOKEN   # no real BOS; EOS as neutral start

    def encode_special(self, token_str: str) -> int:
        # Engine.generate_batch calls this for "<|assistant_end|>".
        # Return EOS_TOKEN so generation stops when model emits EOS.
        return EOS_TOKEN

    def encode(self, texts_or_paths,
               prepend: Optional[Union[int, str]] = None,
               num_threads: int = 1) -> List[List[int]]:
        bos = (self.get_bos_token_id()
               if prepend in ("<|bos|>", EOS_TOKEN)
               else (prepend if isinstance(prepend, int) else None))
        out = []
        for item in (texts_or_paths or []):
            try:
                lookup_dict = pickle.load(open(os.path.join(DEFAULT_CACHE,'note_lookup.pkl'), 'rb'))
                toks = midi_to_tokens(str(item), lookup_dict)
            except Exception:
                toks = [EOS_TOKEN]
            if bos is not None:
                toks = [bos] + toks
            out.append(toks)
        return out

    def decode(self, tokens) -> str:
        return f"<midi:{len(tokens)}_tokens>"

    def __call__(self, text_or_path, prepend=None):
        return self.encode([text_or_path], prepend=prepend)[0]

    def __len__(self):
        return VOCAB_SIZE


HuggingFaceTokenizer = MidiTokenizer   # alias for loss_eval.py / core_eval.py


def get_tokenizer(base_dir=None) -> MidiTokenizer:
    return MidiTokenizer()


def get_token_bytes(tokenizer=None, device=None) -> dict:
    """get_token_bytes(device=device) — accepts and ignores both kwargs."""
    return {i: 1.0 for i in range(VOCAB_SIZE)}
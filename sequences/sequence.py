from __future__ import annotations

import numpy as np

# =========================
# Fast DNA encoders (ACGT + unknown)
# =========================

# LUT: bytes -> 0..3 for A,C,G,T, else 4 (unknown)
_MONO_LUT = np.full(256, 4, dtype=np.int16)
_MONO_LUT[ord('A')] = 0
_MONO_LUT[ord('C')] = 1
_MONO_LUT[ord('G')] = 2
_MONO_LUT[ord('T')] = 3
# lowercase
_MONO_LUT[ord('a')] = 0
_MONO_LUT[ord('c')] = 1
_MONO_LUT[ord('g')] = 2
_MONO_LUT[ord('t')] = 3

# GC indicator: C/G -> 1 else 0 (unknown treated as 0)
_GC_LUT = np.zeros(256, dtype=np.int8)
_GC_LUT[ord('C')] = 1
_GC_LUT[ord('G')] = 1
# lowercase
_GC_LUT[ord('c')] = 1
_GC_LUT[ord('g')] = 1

def encode_sequence_mono(seq: str) -> np.ndarray:
    """Encode bases to int16 array in {0,1,2,3,4} (unknown=4)."""
    b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _MONO_LUT[b].astype(np.int16, copy=False)


def encode_sequence_di(seq: str) -> np.ndarray:
    """
    Returns dinucleotide codes in 0..15 (16 symbols) for positions 1..T-1.
    Code = prev*4 + curr.
    Unknown handling: if either base is unknown -> 16 (unknown dinuc symbol).
    """
    mono = encode_sequence_mono(seq)  # (T,)
    if mono.size < 2:
        return np.zeros(0, dtype=np.int16)

    prev = mono[:-1]
    curr = mono[1:]
    out = (prev * 4 + curr).astype(np.int16, copy=False)

    unk = (prev >= 4) | (curr >= 4)
    if np.any(unk):
        out = out.copy()
        out[unk] = 16
    return out

# =========================
# Labels + augmentation utilities
# =========================

def encode_labels(lbl: str, state_to_idx: dict) -> np.ndarray:
    """
    lbl is a string over label alphabet (e.g. 'N'/'C' in baseline).
    Multi-state training is derived from these labels by the StateScheme.
    """
    try:
        return np.fromiter((state_to_idx[c] for c in lbl), dtype=np.int16, count=len(lbl))
    except KeyError as e:
        raise ValueError(f"Unexpected label char in training labels: {e}") from e


_COMP = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C"})

def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]

def reverse_labels(lbl: str) -> str:
    return lbl[::-1]

def augment_training_data(train_pairs):
    out = list(train_pairs)
    for seq, lbl in train_pairs:
        out.append((revcomp(seq), lbl[::-1]))
    return out
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from sequences.sequence import (
    encode_sequence_mono,
    encode_sequence_di,
)


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    encode: Callable[[str], np.ndarray]
    vocab_size: int
    label_offset: int

    def align_labels_bool(self, true_is_c: np.ndarray, obs_len: int) -> Tuple[np.ndarray, int, int]:
        """
        Align a per-base boolean label array to observation positions.

        Returns:
          (true_obs_is_c, prefix_len, prefix_c)

        - true_obs_is_c: bool array of length obs_len, aligned with observation indices.
        - prefix_len: == label_offset
        - prefix_c: number of True ('C') in the prefix that gets dropped by label_offset
        """
        off = int(self.label_offset)
        if off < 0:
            raise ValueError("label_offset must be >= 0")

        if true_is_c.dtype != np.bool_:
            true_is_c = true_is_c.astype(np.bool_, copy=False)

        prefix_len = off
        if off == 0:
            aligned = true_is_c[:obs_len]
            prefix_c = 0
        else:
            prefix = true_is_c[:off]
            prefix_c = int(prefix.sum())
            aligned = true_is_c[off:off + obs_len]

        if aligned.shape[0] != obs_len:
            raise ValueError(
                f"align_labels_bool mismatch: obs_len={obs_len} but aligned_len={aligned.shape[0]} "
                f"(label_offset={off}, true_len={true_is_c.shape[0]})."
            )
        return aligned, prefix_len, prefix_c

def encode_sequence_mono_offset2(seq: str) -> np.ndarray:
    # assumes you already have fast mono encoding; we just slice from base index 2 onward
    x = encode_sequence_mono(seq)
    return x[2:]

def encode_sequence_mono_offset3(seq: str) -> np.ndarray:
    x = encode_sequence_mono(seq)
    return x[3:]


# Baseline encoders
MONO = EncoderSpec(name="mono", encode=encode_sequence_mono, vocab_size=4,  label_offset=0)
DI = EncoderSpec(name="di",   encode=encode_sequence_di,   vocab_size=17, label_offset=1)
MONO3 = EncoderSpec(name="mono3", encode=encode_sequence_mono_offset2, vocab_size=4, label_offset=2)
MONO4 = EncoderSpec(name="mono4", encode=encode_sequence_mono_offset3, vocab_size=4, label_offset=3)

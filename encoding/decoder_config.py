from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

DecodeMode = Literal["viterbi", "posterior"]

@dataclass(frozen=True)
class DecoderConfig:
    mode: DecodeMode = "viterbi"
    min_cpg_len: int = 0
    tau: Optional[float] = None  # if set and mode='posterior', use this fixed tau (no tuning)
    switch_penalty: float = 0.0

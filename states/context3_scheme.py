from __future__ import annotations

import numpy as np
from itertools import product

from states.states import StateScheme


_BASES = ("A", "C", "G", "T")
_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


class Context3Scheme(StateScheme):
    """
    3rd-order context scheme:
      state = (region ∈ {C,N}, b_{t-2}, b_{t-1}, b_t)

    Total states = 2 * 4^3 = 128.
    """

    def __init__(self):
        states: list[str] = []
        is_cpg: list[bool] = []

        for region in ("C", "N"):
            for kmer in product(_BASES, repeat=3):
                states.append(f"{region}|{kmer[0]}{kmer[1]}{kmer[2]}")
                is_cpg.append(region == "C")

        super().__init__(
            states=states,
            is_cpg_state=np.array(is_cpg, dtype=bool),
        )

    def labels_to_states(self, lbl: str, *, seq=None) -> np.ndarray:
        if seq is None:
            raise ValueError("Context3Scheme requires seq for labels_to_states")

        T = len(lbl)
        if len(seq) != T:
            raise ValueError("seq and lbl must have same length")

        y = np.zeros(T, dtype=np.int16)

        # y[0], y[1] are dummy; trainer slices with obs_offset=2
        if T < 3:
            return y

        for t in range(2, T):
            r = lbl[t]
            a, b, c = seq[t - 2], seq[t - 1], seq[t]

            if a not in _BASE_TO_IDX or b not in _BASE_TO_IDX or c not in _BASE_TO_IDX:
                a = b = c = "A"

            k = (_BASE_TO_IDX[a] * 16) + (_BASE_TO_IDX[b] * 4) + _BASE_TO_IDX[c]  # 0..63
            region_bit = 0 if r == "C" else 1
            y[t] = region_bit * 64 + k  # 0..127

        return y

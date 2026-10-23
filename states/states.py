from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StateScheme:
    states: list[str]
    is_cpg_state: np.ndarray  # shape (K,), bool

    def __post_init__(self):
        if self.is_cpg_state.dtype != bool:
            raise TypeError("is_cpg_state must be boolean")
        if self.is_cpg_state.shape != (len(self.states),):
            raise ValueError("is_cpg_state shape mismatch")

    @property
    def K(self) -> int:
        return len(self.states)

    @property
    def state_to_idx(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.states)}

    def labels_to_states(self, lbl: str, *, seq=None) -> np.ndarray:
        """Convert label string from training file into integer state indices."""
        raise NotImplementedError

    def path_to_output(self, path: np.ndarray) -> str:
        """Convert decoded state indices to 'C'/'N' output string."""
        out = np.where(self.is_cpg_state[path], "C", "N")
        return "".join(out.tolist())


class TwoStateScheme(StateScheme):
    def __init__(self):
        super().__init__(states=["N", "C"], is_cpg_state=np.array([False, True], dtype=bool))

    def labels_to_states(self, lbl: str, seq=None) -> np.ndarray:
        # training labels are exactly 'N'/'C'
        lut = self.state_to_idx  # {'N':0,'C':1}
        try:
            return np.fromiter((lut[c] for c in lbl), dtype=np.int16, count=len(lbl))
        except KeyError as e:
            raise ValueError(f"Unexpected label char {e} in training labels") from e

class FourStateScheme(StateScheme):
    """
    States: N, C_enter, C_core, C_exit
    Training label file is still only N/C, so we derive substates by splitting each C-run.
    """
    def __init__(self):
        states = ["N", "C_enter", "C_core", "C_exit"]
        is_cpg = np.array([False, True, True, True], dtype=bool)
        super().__init__(states=states, is_cpg_state=is_cpg)

    def labels_to_states(self, lbl: str, seq=None) -> np.ndarray:
        # Vectorized-ish approach (O(T) but no per-position Python loop).
        # We detect C-runs and then fill indices.
        b = np.frombuffer(lbl.encode("ascii"), dtype=np.uint8)
        T = b.size
        y = np.zeros(T, dtype=np.int16)  # default N=0

        C = ord("C")
        isC = (b == C)
        if not np.any(isC):
            return y

        # Find run starts/ends
        # start: isC[t] and (t==0 or not isC[t-1])
        # end:   isC[t] and (t==T-1 or not isC[t+1])
        prev = np.r_[False, isC[:-1]]
        nxt = np.r_[isC[1:], False]
        starts = np.where(isC & ~prev)[0]
        ends = np.where(isC & ~nxt)[0]

        # Fill each run: still loops over runs, but runs are few (<= number of islands)
        for s, e in zip(starts, ends):
            if s == e:
                y[s] = 2              # C_core for length-1 run
            else:
                y[s] = 1              # enter
                if e - s > 1:
                    y[s+1:e] = 2      # core
                y[e] = 3              # exit
        return y

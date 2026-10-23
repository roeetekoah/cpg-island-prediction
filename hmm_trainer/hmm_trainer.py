from __future__ import annotations

import numpy as np
from hmm_trainer.theta import Theta, LogTheta


class HMMTrainer:
    """
    Supervised, counts-based HMM trainer.

    Encoder contract:
      - old style: encoder(seq) -> x (obs codes), shape (T,)
      - new style: encoder(seq) -> (x, g), where:
            x: obs codes, shape (T,) with values in [0, V) or out-of-range for unknown
            g: GC flag aligned to x, shape (T,) with values in {0,1}
    """
    def __init__(self, train, states, alpha=1e-3, *, encoder, num_chars: int, obs_offset: int, trans_epsilon: float = 0.0, trans_diag_bonus: float = 0.0) -> None:
        self._encoder = encoder
        self._num_chars = num_chars
        self._obs_offset = obs_offset

        self._trans_epsilon = float(trans_epsilon)
        self._trans_diag_bonus = float(trans_diag_bonus)

        self._theta = None
        self._train = train
        self._alpha = alpha

        self.states = states

        self._init_cnt = np.zeros(self.num_states, dtype=np.float64)
        self._trans_cnt = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self._emit_cnt = np.zeros((self.num_states, self.num_chars), dtype=np.float64)

        # Optional GC feature counts: (K,2)
        self._gc_cnt = np.zeros((self.num_states, 2), dtype=np.float64)
        self._use_gc = False

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_chars(self) -> int:
        return self._num_chars

    @property
    def alpha(self) -> float:
        return self._alpha

    def train(self):
        for seq, lbl in self._train:
            self.update_counts(seq, lbl)

        self.normalize_and_smooth()

    def update_counts(self, seq: str, y: np.ndarray) -> None:
        if len(seq) == 0:
            return

        enc_out = self._encoder(seq)

        if isinstance(enc_out, tuple):
            x, g = enc_out
            x = np.asarray(x)
            g = np.asarray(g)
            if x.shape != g.shape:
                raise ValueError(f"Encoder returned x,g with different shapes: {x.shape} vs {g.shape}")
            self._use_gc = True
        else:
            x = np.asarray(enc_out)
            g = None

        # obs length: T or T-1
        y_aligned = y[self._obs_offset:]  # label length: T or T-1

        if len(x) != len(y_aligned):
            raise ValueError("Mismatch after alignment: len(x) != len(y_aligned)")

        if len(y_aligned) == 0:
            return

        self._init_cnt[int(y_aligned[0])] += 1.0

        # Emissions for x
        mask = (x >= 0) & (x < self.num_chars)
        np.add.at(self._emit_cnt, (y_aligned[mask], x[mask]), 1.0)

        # Optional GC feature
        if g is not None:
            gmask = mask & ((g == 0) | (g == 1))
            if np.any(gmask):
                np.add.at(self._gc_cnt, (y_aligned[gmask], g[gmask]), 1.0)

        # Transitions
        if len(y_aligned) >= 2:
            np.add.at(self._trans_cnt, (y_aligned[:-1], y_aligned[1:]), 1.0)

    '''def normalize_and_smooth(self):
        K = self.num_states
        V = self.num_chars
        a = self.alpha

        init = (self._init_cnt + a) / (self._init_cnt.sum() + K * a)
        trans = (self._trans_cnt + a) / (self._trans_cnt.sum(axis=1, keepdims=True) + K * a)
        emit = (self._emit_cnt + a) / (self._emit_cnt.sum(axis=1, keepdims=True) + V * a)

        gc_emit = None
        if self._use_gc:
            # symmetric Dirichlet on Bernoulli outcomes
            gc_emit = (self._gc_cnt + a) / (self._gc_cnt.sum(axis=1, keepdims=True) + 2 * a)

        self._theta = Theta(self.states, init, trans, emit, gc_emit=gc_emit)'''

    def normalize_and_smooth(self):
        K = self.num_states
        V = self.num_chars
        a = float(self.alpha)

        # --- init ---
        init_num = self._init_cnt + a
        init_den = init_num.sum()
        if init_den == 0:
            init = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            init = init_num / init_den

        # --- trans (THIS is where "bayesian" bonuses go) ---
        trans_num = self._trans_cnt.astype(np.float64, copy=True)

        # Dirichlet smoothing over next-state choices
        if a > 0:
            trans_num += a

        # "Bayesian-ish" baseline mass everywhere (optional)
        if self._trans_epsilon > 0.0:
            trans_num += self._trans_epsilon

        # "Bayesian-ish" diagonal bonus (optional)
        if self._trans_diag_bonus > 0.0:
            idx = np.arange(K)
            trans_num[idx, idx] += self._trans_diag_bonus

        trans_den = trans_num.sum(axis=1, keepdims=True)
        # avoid divide-by-zero
        trans_den = np.where(trans_den == 0.0, 1.0, trans_den)
        trans = trans_num / trans_den

        # --- emit (same as before) ---
        emit_num = self._emit_cnt + a
        emit_den = emit_num.sum(axis=1, keepdims=True)
        emit_den = np.where(emit_den == 0.0, 1.0, emit_den)
        emit = emit_num / emit_den

        # --- optional GC feature emission (Bernoulli per state) ---
        gc_emit = None
        if self._use_gc:
            gc_num = self._gc_cnt + a
            gc_den = gc_num.sum(axis=1, keepdims=True)
            gc_den = np.where(gc_den == 0.0, 1.0, gc_den)
            gc_emit = gc_num / gc_den

        self._theta = Theta(self.states, init, trans, emit, gc_emit=gc_emit)

    @property
    def theta(self) -> Theta:
        if self._theta is None:
            raise RuntimeError("Call train() first.")
        return self._theta

    @property
    def log_theta(self) -> LogTheta:
        return LogTheta.from_theta(self.theta)

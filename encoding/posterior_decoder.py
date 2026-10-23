from __future__ import annotations
import numpy as np
import math

from hmm_trainer.theta import LogTheta
from encoding.encoder import EncoderSpec


def logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)
    out = m_safe + np.log(s)
    return np.squeeze(out, axis=axis)


class PosteriorDecoder:
    """
    Forward–Backward posterior decoder.
    Produces base-level labels via tau thresholding.
    Supports optional factorized GC feature.
    """

    def __init__(self, log_theta: LogTheta, *, enc: EncoderSpec, is_cpg_state: np.ndarray):
        self.lt = log_theta
        self.enc = enc
        self.is_cpg_state = is_cpg_state

        self.K = len(log_theta.states)
        self.V = enc.vocab_size

        self._log_uniform_emit = -math.log(self.V)
        self._log_uniform_gc = -math.log(2.0)

        if self.lt.gc_emit is not None and self.lt.gc_emit.shape != (self.K, 2):
            raise ValueError("log_theta.gc_emit must have shape (K,2)")

    def _emissions(self, seq: str) -> np.ndarray:
        """
        Returns E[t,k] = log P(x_t, g_t | state=k)
        """
        enc_out = self.enc.encode(seq)

        if isinstance(enc_out, tuple):
            x, g = enc_out
            x = np.asarray(x, dtype=np.int64)
            g = np.asarray(g, dtype=np.int64)
            if x.shape != g.shape:
                raise ValueError("encoder returned x,g with different shapes")
        else:
            x = np.asarray(enc_out, dtype=np.int64)
            g = None

        T = len(x)
        E = np.empty((T, self.K), dtype=np.float64)

        # ---- base emission ----
        unk = (x < 0) | (x >= self.V)
        xc = x.copy()
        xc[unk] = 0

        E[:, :] = self.lt.emit[:, xc].T
        if np.any(unk):
            E[unk, :] = self._log_uniform_emit

        # ---- factorized GC emission ----
        if self.lt.gc_emit is not None and g is not None:
            gunk = (g != 0) & (g != 1)
            gc = g.copy()
            gc[gunk] = 0

            Egc = self.lt.gc_emit[:, gc].T
            if np.any(gunk):
                Egc[gunk, :] = self._log_uniform_gc

            E += Egc

        return E

    def cpg_posterior(self, seq: str) -> np.ndarray:
        E = self._emissions(seq)
        T = E.shape[0]

        log_alpha = np.empty((T, self.K))
        log_beta = np.empty((T, self.K))

        log_alpha[0] = self.lt.init + E[0]
        for t in range(1, T):
            log_alpha[t] = E[t] + logsumexp(
                log_alpha[t - 1][:, None] + self.lt.trans, axis=0
            )

        log_px = logsumexp(log_alpha[T - 1], axis=0)

        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                self.lt.trans
                + E[t + 1][None, :]
                + log_beta[t + 1][None, :],
                axis=1,
            )

        log_post = log_alpha + log_beta - log_px
        log_p_cpg = logsumexp(log_post[:, self.is_cpg_state], axis=1)
        return np.exp(log_p_cpg)

    def predict_labels(self, seq: str, tau: float) -> str:
        p = self.cpg_posterior(seq)
        obs_pred = "".join("C" if v >= tau else "N" for v in p)

        if self.enc.label_offset == 0:
            return obs_pred
        return ("N" * self.enc.label_offset) + obs_pred

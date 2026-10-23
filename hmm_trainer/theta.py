from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Theta:
    states: list[str]        # length K
    init: np.ndarray         # shape (K,)
    trans: np.ndarray        # shape (K,K)
    emit: np.ndarray         # shape (K,V)
    gc_emit: np.ndarray | None = None   # shape (K,2) if present


@dataclass(frozen=True)
class LogTheta:
    states: list[str]
    init: np.ndarray         # log π, shape (K,)
    trans: np.ndarray        # log τ, shape (K,K)
    emit: np.ndarray         # log e, shape (K,V)
    gc_emit: np.ndarray | None = None   # log Bernoulli probs, shape (K,2) if present

    @staticmethod
    def from_theta(theta: Theta) -> "LogTheta":
        return LogTheta(
            states=theta.states,
            init=np.log(theta.init),
            trans=np.log(theta.trans),
            emit=np.log(theta.emit),
            gc_emit=None if theta.gc_emit is None else np.log(theta.gc_emit),
        )

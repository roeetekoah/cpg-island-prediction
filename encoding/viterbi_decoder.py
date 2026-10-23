import numpy as np

class ViterbiDecoder:
    def __init__(self, lt, *, encoder, num_chars: int, trans_bias=None):
        self.lt = lt
        self.encoder = encoder
        self.num_chars = num_chars
        self._trans_bias = trans_bias

        # define K,V first (IMPORTANT)
        self._K = int(self.lt.trans.shape[0])
        # prefer actual emission width if present; otherwise use num_chars
        self._V = int(getattr(self.lt, "emit", np.empty((self._K, num_chars))).shape[1])

        # unknown emission vector (per-state), used when obs_id out of range
        self._log_emit_unknown = np.full(self._K, -1e9, dtype=np.float64)

        # Precompute indices of states whose "last base" is A/C/G/T (for context schemes)
        states = getattr(lt, "states", None)
        if states is None:
            self._states_ending_in = None
        else:
            self._states_ending_in = {b: [] for b in "ACGT"}
            for i, s in enumerate(states):
                last = s[-1]
                if last in self._states_ending_in:
                    self._states_ending_in[last].append(i)
            for b in "ACGT":
                self._states_ending_in[b] = np.array(self._states_ending_in[b], dtype=np.int32)

    def _emit_col(self, obs_id: int) -> np.ndarray:
        """Return log emission vector for obs_id; fallback if out-of-range."""
        if 0 <= obs_id < self._V:
            return self.lt.emit[:, obs_id]
        return self._log_emit_unknown

    def decode(self, seq: str) -> np.ndarray:
        x = self.encoder(seq)
        T = len(x)
        K = self._K

        trans = self.lt.trans if self._trans_bias is None else (self.lt.trans + self._trans_bias)

        dp = np.full((T, K), -np.inf, dtype=np.float64)
        back = np.full((T, K), -1, dtype=np.int32)

        idx_to_base = {0: "A", 1: "C", 2: "G", 3: "T"}

        # init (SAFE)
        obs0 = int(x[0])
        dp[0, :] = self.lt.init + self._emit_col(obs0)

        # pruning init
        if self._states_ending_in is not None:
            b0 = idx_to_base.get(obs0, None)  # None if unknown / out of 0..3
            if b0 in self._states_ending_in:
                active_prev = self._states_ending_in[b0]
                mask = np.ones(K, dtype=bool)
                mask[active_prev] = False
                dp[0, mask] = -np.inf
            else:
                active_prev = np.arange(K, dtype=np.int32)  # unknown -> no pruning
        else:
            active_prev = np.arange(K, dtype=np.int32)

        for t in range(1, T):
            obs = int(x[t])

            # choose active_curr
            if self._states_ending_in is not None:
                bt = idx_to_base.get(obs, None)
                if bt is None:
                    active_curr = np.arange(K, dtype=np.int32)
                else:
                    active_curr = self._states_ending_in[bt]
            else:
                active_curr = np.arange(K, dtype=np.int32)

            # Dense fallback if pruning gives empty sets
            if active_prev.size == 0 or active_curr.size == 0:
                scores = dp[t - 1][:, None] + trans
                best_prev = np.argmax(scores, axis=0)
                dp[t] = np.max(scores, axis=0) + self._emit_col(obs)   # SAFE
                back[t] = best_prev
                active_prev = np.arange(K, dtype=np.int32)
                continue

            # PRUNED step
            prev_vals = dp[t - 1, active_prev][:, None]
            sub_trans = trans[np.ix_(active_prev, active_curr)]
            scores = prev_vals + sub_trans

            arg = np.argmax(scores, axis=0)
            best_prev_abs = active_prev[arg]
            best_score = scores[arg, np.arange(scores.shape[1])]

            # add emissions safely for active_curr only
            if 0 <= obs < self._V:
                emit_curr = self.lt.emit[active_curr, obs]
            else:
                emit_curr = self._log_emit_unknown[active_curr]

            dp[t, active_curr] = best_score + emit_curr
            back[t, active_curr] = best_prev_abs
            active_prev = active_curr

        # backtrack
        path = np.zeros(T, dtype=np.int32)
        path[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
            if path[t] < 0:
                path[t] = path[t + 1]
        return path

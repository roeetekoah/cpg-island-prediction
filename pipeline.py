from __future__ import annotations

import os
from dataclasses import dataclass
import gzip
import numpy as np
import time

from sequences.fasta import parse_fasta_file
from hmm_trainer.hmm_trainer import HMMTrainer
from hmm_trainer.theta import LogTheta
from validation.evaluate import Metrics, evaluate_predictions, metrics_from_counts
from sequences.sequence import augment_training_data
from states.states import StateScheme
from encoding.encoder import EncoderSpec
from encoding.decoder_config import DecoderConfig
from encoding.viterbi_decoder import ViterbiDecoder
from encoding.posterior_decoder import PosteriorDecoder
from encoding.postprocess import remove_short_cpg_segments

@dataclass(frozen=True)
class Experiment:
    name: str
    scheme: StateScheme
    encoder: EncoderSpec
    use_revcomp: bool
    alpha: float
    trans_epsilon: float = 0.0
    trans_diag_bonus: float = 0.0


class CPGPipeline:
    def __init__(self, *, seed: int = 0, verbose: bool = False, log_every: int = 200):
        self.seed = seed
        self.verbose = verbose
        self.log_every = log_every

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return

        print(msg)

    def split_pairs(self, pairs, train_frac=0.7, val_frac=0.15):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(pairs))
        rng.shuffle(idx)
        n = len(pairs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = [pairs[i] for i in idx[:n_train]]
        val = [pairs[i] for i in idx[n_train:n_train + n_val]]
        test = [pairs[i] for i in idx[n_train + n_val:]]
        return train, val, test

    def train(self, exp: Experiment, pairs: list[tuple[str, str]]) -> LogTheta:
        t0 = time.perf_counter()
        self._log(
            f"TRAIN start: [{exp.name}] alpha={exp.alpha:g} revcomp={exp.use_revcomp} "
            f"states={len(exp.scheme.states)} enc={exp.encoder.name} n_pairs={len(pairs)}"
        )

        if exp.use_revcomp:
            pairs = augment_training_data(pairs)

        train_for_trainer = [(seq, exp.scheme.labels_to_states(lbl, seq=seq)) for seq, lbl in pairs]

        trainer = HMMTrainer(
            train_for_trainer,
            exp.scheme.states,
            alpha=exp.alpha,
            encoder=exp.encoder.encode,
            num_chars=exp.encoder.vocab_size,
            obs_offset=exp.encoder.label_offset,
            trans_epsilon=exp.trans_epsilon,
            trans_diag_bonus=exp.trans_diag_bonus
        )

        trainer.train()

        dt = time.perf_counter() - t0
        self._log(f"TRAIN done : [{exp.name}] time={dt:.2f}s")
        return trainer.log_theta

    @staticmethod
    def build_switch_penalty_bias(is_cpg_state: np.ndarray, lam: float) -> np.ndarray:
        """
        Returns bias matrix B where:
          B[i,j] = -lam if region(i) != region(j) else 0
        """
        if lam <= 0:
            return np.zeros((len(is_cpg_state), len(is_cpg_state)), dtype=np.float64)

        same_region = (is_cpg_state[:, None] == is_cpg_state[None, :])
        B = np.zeros((len(is_cpg_state), len(is_cpg_state)), dtype=np.float64)
        B[~same_region] = -lam
        return B

    def _predict_labels(
            self,
            lt: LogTheta,
            exp: Experiment,
            seq: str,
            cfg: DecoderConfig,
            *,
            tau: float | None,
    ) -> str:
        """
        Returns BASE-length label string ('C'/'N') aligned to original DNA length.
        - mono: obs length == base length
        - di:   obs length == base length - 1, so we prefix 'N' once
        """
        if cfg.mode == "viterbi":
            trans_bias = None
            if cfg.switch_penalty and cfg.switch_penalty > 0:
                trans_bias = CPGPipeline.build_switch_penalty_bias(exp.scheme.is_cpg_state, float(cfg.switch_penalty))

            dec = ViterbiDecoder(
                lt,
                encoder=exp.encoder.encode,
                num_chars=exp.encoder.vocab_size,
                trans_bias=trans_bias,
            )

            path = dec.decode(seq)
            obs_pred = exp.scheme.path_to_output(path)  # length = obs length
            if exp.encoder.label_offset == 0:
                base_pred = obs_pred
            else:
                base_pred = ("N" * exp.encoder.label_offset) + obs_pred

        elif cfg.mode == "posterior":
            if tau is None:
                raise ValueError("tau must be provided for posterior decoding")
            dec = PosteriorDecoder(lt, enc=exp.encoder, is_cpg_state=exp.scheme.is_cpg_state)
            base_pred = dec.predict_labels(seq, tau=tau)

        else:
            raise ValueError(f"Unknown decoder mode: {cfg.mode}")

        # Post-process on BASE-length string
        if cfg.min_cpg_len and cfg.min_cpg_len > 1:
            base_pred = remove_short_cpg_segments(base_pred, cfg.min_cpg_len)

        return base_pred

    def eval(
        self,
        lt: LogTheta,
        exp: Experiment,
        pairs: list[tuple[str, str]],
        cfg: DecoderConfig,
        *,
        tau: float | None,
    ) -> Metrics:
        TP = FP = FN = TN = 0
        t0 = time.perf_counter()

        for i, (seq, true_lbl) in enumerate(pairs, 1):
            pred_lbl = self._predict_labels(lt, exp, seq, cfg, tau=tau)
            tp, fp, fn, tn = evaluate_predictions(true_lbl, pred_lbl)
            TP += tp;
            FP += fp;
            FN += fn;
            TN += tn

            if self.verbose and (i % self.log_every == 0):
                dt = time.perf_counter() - t0
                avg = dt / i
                self._log(
                    f"EVAL progress: [{exp.name}] {i}/{len(pairs)} seqs, "
                    f"elapsed={dt:.1f}s avg={avg:.3f}s/seq last_len={len(seq)}"
                )

        return metrics_from_counts(TP, FP, FN, TN)

    def annotate_fasta(
            self,
            lt: LogTheta,
            exp: Experiment,
            cfg: DecoderConfig,
            *,
            tau: float | None,
            input_path: str,
            output_path: str,
    ):
        t0 = time.perf_counter()
        self._log(f"ANNOTATE start: [{exp.name}] mode={cfg.mode} input={input_path} output={output_path}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        seqs = parse_fasta_file(input_path)
        with gzip.open(output_path, "wt") as f:
            for i, (seq_id, seq) in enumerate(seqs.items(), 1):
                ann = self._predict_labels(lt, exp, seq, cfg, tau=tau)
                f.write(f">{seq_id}\n{ann}\n")
                if self.verbose and (i % self.log_every == 0):
                    self._log(f"  annotated {i} sequences...")

        dt = time.perf_counter() - t0
        self._log(f"ANNOTATE done : wrote {len(seqs)} sequences, time={dt:.2f}s")

    def annotate_sequence(self, lt: LogTheta, exp: Experiment, cfg: DecoderConfig, sequence: str, *, tau: float = None) -> str:
        return self._predict_labels(lt, exp, sequence, cfg, tau=tau)

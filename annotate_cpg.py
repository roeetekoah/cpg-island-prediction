# annotate_cpg.py
from __future__ import annotations

import argparse
import os
import time

from hmm_trainer.theta import LogTheta
from sequences.fasta import prepare_training_data
from states.context3_scheme import Context3Scheme
from encoding.encoder import MONO3
from pipeline import CPGPipeline, Experiment
from encoding.decoder_config import DecoderConfig

from dataclasses import dataclass

@dataclass
class TrainedCpGModel:
    lt: LogTheta          # your trained LogTheta / model params
    exp: Experiment     # experiment config (scheme, encoder, alpha, revcomp flag, etc.)
    cfg: DecoderConfig  # decode config (viterbi/posterior, min len, etc.)
    pipe: CPGPipeline   # pipeline object (for decoding helpers, etc.)


def train_classifier(training_data, verbose: bool = False):
    """
    REQUIRED API.
    training_data: list[tuple[str, str]]  (sequence, label)
    returns: model object
    """
    pipe = CPGPipeline(seed=0, verbose=verbose, log_every=200)
    exp = Experiment("final", Context3Scheme(), MONO3, use_revcomp=True, alpha=1e-10)
    cfg = DecoderConfig(mode="viterbi", min_cpg_len=50)
    lt = pipe.train(exp, training_data)

    return TrainedCpGModel(lt=lt, exp=exp, cfg=cfg, pipe=pipe)


def annotate_sequence(model, sequence: str) -> str:
    """
    REQUIRED API.
    returns a string of 'C' and 'N' same length as sequence.
    """
    return model.pipe.annotate_sequence(model.lt, model.exp, model.cfg, sequence, tau=None)


def build_base_experiments(alpha: float) -> list[Experiment]:
    return [
        Experiment("128state_aug", Context3Scheme(), MONO3, use_revcomp=False, alpha=float(alpha)),
    ]


def _parse_float_list(s: str) -> list[float]:
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",")]

def _build_run_objects(verbose: bool):
    pipe = CPGPipeline(seed=0, verbose=verbose, log_every=200)
    exp = Experiment("final", Context3Scheme(), MONO3, use_revcomp=True, alpha=1e-10)
    cfg = DecoderConfig(mode="viterbi", min_cpg_len=50, tau=0.7)
    tau = (cfg.tau if cfg.mode == "posterior" else None)
    return pipe, exp, cfg, tau

def _print_run_banner(*, exp, cfg, tau, stage: str) -> None:
    print(f"[CpG] === {stage} CONFIG ===")
    print(f"[CpG] experiment   : {exp.name}")
    print(f"[CpG] states       : {len(exp.scheme.states)}")
    print(f"[CpG] encoder      : {exp.encoder.name}  (vocab={exp.encoder.vocab_size}, offset={exp.encoder.label_offset})")
    print(f"[CpG] revcomp aug  : {exp.use_revcomp}")
    print(f"[CpG] alpha        : {exp.alpha:g}")
    print(f"[CpG] decode       : {cfg.mode}")
    if cfg.mode == "posterior":
        print(f"[CpG] tau          : {tau}")
    print(f"[CpG] min_cpg_len  : {cfg.min_cpg_len}")
    # only if you have it in DecoderConfig
    if getattr(cfg, "switch_penalty", None) is not None:
        print(f"[CpG] switch_penalty: {cfg.switch_penalty}")
    print(f"[CpG] =====================")

def main():
    p = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    p.add_argument("--training-sequences", required=True, help="Labeled training DNA FASTA (.fa or .fa.gz).")
    p.add_argument("--training-labels", required=True, help="Matching per-base C/N label FASTA.")
    p.add_argument("--fasta_path", type=str, required=False)
    p.add_argument("--output_file", type=str, required=False)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--no_annotate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    t_total0 = time.perf_counter()
    print("[CpG] Starting run")

    training_sequences_path = args.training_sequences
    training_labels_path = args.training_labels

    print("[CpG] Loading training data...")
    raw_pairs = prepare_training_data(training_sequences_path, training_labels_path)
    print(f"[CpG] Loaded {len(raw_pairs)} labeled sequences")

    pipe, exp, cfg, tau = _build_run_objects(args.verbose)

    # decide data splits once
    if args.eval:
        train_pairs, val_pairs, test_pairs = pipe.split_pairs(raw_pairs)
        train_used = train_pairs
        print(f"[CpG] Split: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    else:
        train_pairs, val_pairs, test_pairs = raw_pairs, None, None
        train_used = raw_pairs
        print(f"[CpG] Train on ALL data: {len(train_used)} sequences")

    # print plan BEFORE doing anything heavy
    _print_run_banner(exp=exp, cfg=cfg, tau=tau, stage="PLAN")

    # train once
    t0 = time.perf_counter()
    lt = pipe.train(exp, train_used)
    t_train = time.perf_counter() - t0
    print(f"[CpG] Train time: {t_train:.2f}s")

    # create model object for required API compatibility
    model = TrainedCpGModel(lt=lt, exp=exp, cfg=cfg, pipe=pipe)

    # eval if requested
    if args.eval:
        t1 = time.perf_counter()
        m_val = pipe.eval(lt, exp, val_pairs, cfg, tau=tau)
        t_val = time.perf_counter() - t1

        t2 = time.perf_counter()
        m_test = pipe.eval(lt, exp, test_pairs, cfg, tau=tau)
        t_test = time.perf_counter() - t2

        print(f"[CpG] VAL : acc={m_val.accuracy:.4f}  F1(C)={m_val.f1_c:.4f}  "
              f"P={m_val.precision_c:.4f}  R={m_val.recall_c:.4f}  time={t_val:.2f}s")
        print(f"[CpG] TEST: acc={m_test.accuracy:.4f}  F1(C)={m_test.f1_c:.4f}  "
              f"P={m_test.precision_c:.4f}  R={m_test.recall_c:.4f}  time={t_test:.2f}s")

    # annotate if requested
    if (not args.no_annotate) and args.fasta_path and args.output_file:
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"[CpG] Annotating {args.fasta_path}")
        print(f"[CpG] Writing   {args.output_file}")

        t3 = time.perf_counter()
        pipe.annotate_fasta(
            lt, exp, cfg,
            tau=tau,
            input_path=args.fasta_path,
            output_path=args.output_file,
        )
        t_annot = time.perf_counter() - t3
        print(f"[CpG] Annotate time: {t_annot:.2f}s")
    else:
        print("[CpG] Annotation skipped")

    print(f"[CpG] Total runtime: {time.perf_counter() - t_total0:.2f}s")


if __name__ == "__main__":
    main()

# CpG Island Prediction with Hidden Markov Models

**From DNA sequence to per-base CpG-island annotations.** A computational-biology coursework implementation of supervised HMM training, context-expanded sequence models, and two approaches to decoding.

The project explores how sequence context, reverse-complement augmentation, and minimum island length affect annotation. Its modular pipeline separates state design, DNA encoding, parameter estimation, decoding, and evaluation, making those choices explicit rather than hiding them behind a classifier API.

## What is implemented

- Counts-based estimation of initial, transition, and emission probabilities with smoothing.
- 128-state and 512-state nucleotide-context models.
- Log-space Viterbi and forward–backward posterior decoding.
- Reverse-complement augmentation, short-island filtering, and per-base precision, recall, and F1.
- FASTA input and gzipped annotation output; a seeded train/validation/test split for supplied labeled data.

The default classifier uses 128 context states, reverse-complement augmentation, Viterbi decoding, and a 50-base minimum island length.

## Try the pipeline

Use Python 3.10 or newer from the repository root:

```sh
python -m pip install -r requirements.txt
python run_demo.py
```

The short demo trains on an explicitly **synthetic fixture**, runs both decoders, checks annotation length and alphabet, and verifies the gzipped FASTA round trip. It demonstrates the software path; its predictions are not biological evaluation results.

For real labeled DNA, see the [usage guide](docs/USAGE.md). The original course dataset is not bundled.

## Existing experiment artifacts

[![Historical comparison of CpG-class F1 across configurations](docs/figures/part2_f1_by_config.png)](docs/RESULTS.md)

This original figure records a coursework comparison of model and decoder configurations. It is retained as a historical artifact, not a freshly reproduced benchmark. The [results note](docs/RESULTS.md) explains the edition boundary and includes the island-length comparison.

**Read further:** [Model and repository map](docs/METHOD.md) · [Data, scaffolding, and source provenance](SOURCES.md).

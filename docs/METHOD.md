# Model and repository map

Training is supervised: each DNA sequence is paired with an equal-length string of `C` (CpG island) and `N` (background) labels. The trainer counts labeled state transitions and emissions and then normalizes with additive smoothing. This package does not train the classifier with Baum–Welch or claim to discover biological truth from unlabeled data.

| Component | Implementation |
|---|---|
| State and label mapping | `states/states.py`, `states/context3_scheme.py`, `states/context4_scheme.py` |
| Mononucleotide/dinucleotide observations and context offsets | `encoding/encoder.py`, `sequences/sequence.py` |
| Count estimation and probability/log-probability parameters | `hmm_trainer/` |
| Viterbi path decoding | `encoding/viterbi_decoder.py` |
| Forward–backward posterior decoding | `encoding/posterior_decoder.py` |
| Minimum-length island filter | `encoding/postprocess.py` |
| Training, split, evaluation, and annotation orchestration | `pipeline.py` |
| Fixed default classifier and command line | `annotate_cpg.py` |

The context schemes name states using both region and nucleotide context: 2 × 4³ = 128 states, or 2 × 4⁴ = 512. These are implemented state expansions, not new HMM inference algorithms. Context encoding drops the first two or three observations; annotation restores that prefix as background `N`.

Viterbi selects a state path. Posterior decoding instead thresholds each position's marginal CpG probability. The default command line uses Viterbi; the synthetic demo also exercises posterior decoding with a fixed threshold.

Training data should contain uppercase A/C/G/T and only C/N labels. Use sequences of at least four bases for the included context models. The original implementation's handling of ambiguous bases, empty input, and very short sequences has not been generalized in this packaging pass.

The train/validation/test split is sequence-level and seeded. Reverse-complement augmentation happens after the split when enabled. Evaluation counts bases, not whole islands. This release makes no independent biological-validation or genome-wide generalization claim.

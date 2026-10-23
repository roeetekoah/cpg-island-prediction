# Run on labeled DNA

Install `requirements.txt` with Python 3.10+ and run from the repository root. Every training DNA record needs a label record with the same identifier and length. Labels use `C` for island and `N` for background. FASTA may be plain text or gzip-compressed.

```sh
python annotate_cpg.py --training-sequences data/train.seq.fa --training-labels data/train.lbl.fa --fasta_path data/query.fa --output_file output/query.labels.fa.gz
```

The command trains the fixed default 128-state classifier on all provided pairs and writes gzipped label FASTA. It does not download data or load a hidden pretrained model.

For a seeded 70%/15%/15% sequence split and per-base validation/test metrics:

```sh
python annotate_cpg.py --training-sequences data/train.seq.fa --training-labels data/train.lbl.fa --eval --no_annotate
```

Supply enough labeled sequences for nonempty subsets. This fixed configuration is not a new hyperparameter search, and the historical figures are not guaranteed to be reproduced by this command.

The intentionally artificial example files in `examples/` can also exercise the annotation command:

```sh
python annotate_cpg.py --training-sequences examples/synthetic.seq.fa --training-labels examples/synthetic.lbl.fa --fasta_path examples/synthetic.seq.fa --output_file output/synthetic.labels.fa.gz
```

Those sequences and their labels are hand-constructed for software smoke checks. Do not interpret training-set predictions or printed counts as biological accuracy. Generated outputs are ignored by Git.

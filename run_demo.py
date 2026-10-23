"""Small synthetic software check, deliberately not a biological benchmark."""
from pathlib import Path
import gzip
import tempfile

from annotate_cpg import train_classifier, annotate_sequence
from encoding.decoder_config import DecoderConfig
from sequences.fasta import prepare_training_data, parse_fasta_file


def main():
    root = Path(__file__).resolve().parent
    pairs = prepare_training_data(str(root / "examples/synthetic.seq.fa"),
                                  str(root / "examples/synthetic.lbl.fa"))
    assert len(pairs) == 4
    assert all(len(seq) == len(lbl) and set(seq) <= set("ACGT")
               and set(lbl) <= set("CN") for seq, lbl in pairs)
    model = train_classifier(pairs)
    assert len(model.lt.states) == 128
    for seq, _ in pairs:
        for mode in ("viterbi", "posterior"):
            if mode == "viterbi":
                labels = annotate_sequence(model, seq)
            else:
                labels = model.pipe.annotate_sequence(
                    model.lt, model.exp,
                    DecoderConfig(mode="posterior", min_cpg_len=50),
                    seq, tau=0.5)
            assert len(labels) == len(seq)
            assert set(labels) <= set("CN")
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        output = str(Path(tmp) / "annotations.fa.gz")
        model.pipe.annotate_fasta(model.lt, model.exp, model.cfg, tau=None,
                                  input_path=str(root / "examples/synthetic.seq.fa"),
                                  output_path=output)
        predictions = parse_fasta_file(output)
        assert list(predictions) == ["synthetic_1", "synthetic_2", "synthetic_3", "synthetic_4"]
        assert list(predictions.values()) == [annotate_sequence(model, seq) for seq, _ in pairs]
        with gzip.open(output, "rt") as handle:
            assert handle.read().startswith(">synthetic_1\n")
    print("PASS: synthetic-only smoke check; 4 sequences, 128 states, both decoders, gzip round trip.")
    print("No biological performance claim is made by this demo.")


if __name__ == "__main__":
    main()

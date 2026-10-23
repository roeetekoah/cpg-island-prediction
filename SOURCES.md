# Sources and attribution

This is a curated copy of a computational-biology coursework project from Roee Tekoah's local degree archive. It implements established hidden Markov model methods for a course-supplied CpG annotation task; it does not introduce a new biological predictor or a new inference algorithm.

| Material | Origin and treatment |
|---|---|
| `pipeline.py`, state schemes, encoders, trainer, decoders, evaluation modules | Modular coursework implementation from the archive's Exercise 2, Part 2 directory. Copied without model or inference changes. |
| `annotate_cpg.py` | Same modular edition. Packaging adds explicit training-file arguments instead of assumed `data/` paths and removes an unused import of an incomplete tuning module. |
| `sequences/fasta.py` | FASTA parsing and training-pair preparation supplied in the course's base `annotate_cpg.py`; moved into the original project's module structure. These helpers are credited as course scaffolding, not original algorithmic work. |
| `docs/figures/*.png` | Existing coursework figures from Part 2's `graphs/` directory, copied unchanged. See [results provenance](docs/RESULTS.md). |
| `examples/synthetic.*.fa`, `run_demo.py` | Added for this packaging pass. Hand-constructed sequences and labels for a software smoke check, not course data or experiment results. |
| README and `docs/*.md` | Added for this curated edition. |

The original task supplied labeled DNA and an annotation interface. The biological dataset, lecture slides, assignment PDFs, alternate personal implementations, submission ZIPs, caches, and the older report are not redistributed here. No ownership claim is made over the supplied assignment material or data.

No new license is assigned to the archived coursework or course-supplied scaffolding in this curated edition.

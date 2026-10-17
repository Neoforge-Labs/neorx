"""GenMol validity, measured under each definition of "valid" separately.

Why this exists. README.md claimed "97% validity" for GenMol with no run
record behind it. The claim traces to ``results/genmol_paper4_final.json``,
an untracked output of an untracked script, which measured ONE unseeded
draw at temperature 0.8 from ``checkpoints/genmol_paper4/best_model.pt``
-- the epoch-28 checkpoint, which is not the one the wheel ships. The
shipped ``assets/molvae_chembl36.pt`` is byte-identical to that run's
``final_model.pt`` (epoch 30) instead. So the headline number was measured
on different weights, at a temperature nobody wrote down, with no seed and
no spread.

Four rates, reported separately and never collapsed into one. "Validity"
names at least four different things in the generative-chemistry
literature, and conflating them is how a 97 and an 85 can both be reported
in good faith of the same model:

``parseable``
    RDKit reads the string into a molecule graph at all
    (``sanitize=False``) and that graph has at least one atom. Note that
    ``Chem.MolFromSmiles("")`` returns an empty molecule rather than
    ``None``, so the atom count is part of the question, not a detail.
``sanitisable``
    That graph then survives ``Chem.SanitizeMol`` -- valences,
    aromaticity and kekulisation all mutually consistent. This is the
    strictest structural rate, it is what plain ``Chem.MolFromSmiles(s)
    is not None`` reports, and it is what both the 97% claim and
    ``genmol-eval`` mean by "validity".
``unique``
    Distinct canonical SMILES as a fraction of the sanitisable ones.
``novel``
    Distinct canonical SMILES absent from the training corpus, as a
    fraction of the distinct ones.

Measured as a seed sweep at every temperature the codebase actually uses,
because validity falls steeply with temperature and the README named none:
0.8 is what the original evaluation used, 1.0 is the default of both
``generate()`` and the HTTP API, and 1.2 is what ``core.pipeline`` passes.
A rate quoted without its temperature is not reproducible, and a rate from
a single draw has no error bar -- both failures are what produced the
number this experiment replaces.

The model has documented partial posterior collapse (see
``causalbiorl.envs.generation.decode_latent``), so ``unique`` and ``novel``
are measurements here, not targets.

Offline: loads the shipped weights and samples locally. No HTTP, so no
cassette is recorded. The training corpus is a local prerequisite, not a
packaged asset; ``novel`` cannot be computed without it, so its absence
fails the run rather than quietly dropping the column.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

#: Every temperature the repository actually generates at. 0.8 is
#: experiments/genmol_eval.py, 1.0 is the generate() and api.py default,
#: 1.2 is src/neorx/core/pipeline/__init__.py.
TEMPERATURES = (0.8, 1.0, 1.2)

#: Five seeds, so each rate carries a spread rather than a point. The
#: claim this replaces was a single draw.
SEEDS = (0, 1, 2, 3, 4)

#: Per seed. At 0.8 the binomial standard error on a ~0.95 rate is about
#: 0.003 at this size, well under the spread actually observed between
#: seeds -- so the seed sweep, not the sample size, is the thing that
#: makes the interval honest.
N_PER_SEED = 5000


@experiment(
    name="genmol-validity",
    help="GenMol validity under each definition, across seeds and temperatures.",
    volatile_fields=("generate_s",),
)
def genmol_validity(record: RunRecord) -> None:
    from neorx.genmol import generate, load_pretrained
    from neorx.genmol.data.download import _DEFAULT_CSV, load_smiles

    model, tokenizer = load_pretrained()
    n_params = sum(p.numel() for p in model.parameters())

    # Cite the corpus by content, not by path: `novel` is only meaningful
    # against a named reference set, and this corpus is not a packaged
    # asset, so the digest is the only thing that lets a later reader
    # confirm they are comparing against the same molecules.
    corpus = load_smiles()
    corpus_digest = hashlib.sha256(_DEFAULT_CSV.read_bytes()).hexdigest()
    training = set(corpus)

    for temperature in TEMPERATURES:
        for seed in SEEDS:
            # generate() takes no seed, so the global torch RNG is the only
            # handle on reproducibility. Seeded here rather than once per
            # experiment so each row stands alone: re-running one cell
            # reproduces that cell.
            torch.manual_seed(seed)

            started = time.perf_counter()
            # validate=False, deduplicate=False: generate() otherwise
            # filters internally, which would make every rate below
            # structurally 1.0 -- a better-looking number than the truth
            # rather than a measurement of it.
            smiles = generate(
                model,
                tokenizer,
                n=N_PER_SEED,
                temperature=temperature,
                validate=False,
                deduplicate=False,
            )
            elapsed = time.perf_counter() - started

            counts = _classify(smiles)
            n = len(smiles)
            distinct = counts.distinct
            record.append_row(
                {
                    "n_params": n_params,
                    "vocab_size": tokenizer.vocab_size,
                    "corpus_sha256": corpus_digest,
                    "n_corpus": len(corpus),
                    "temperature": temperature,
                    "seed": seed,
                    "n_requested": N_PER_SEED,
                    "n_generated": n,
                    "n_empty": counts.n_empty,
                    "n_parseable": counts.n_parseable,
                    "n_sanitisable": counts.n_sanitisable,
                    "n_distinct": len(distinct),
                    "n_novel": len(distinct - training),
                    "parseable": counts.n_parseable / n if n else 0.0,
                    "sanitisable": counts.n_sanitisable / n if n else 0.0,
                    "unique": (
                        len(distinct) / counts.n_sanitisable if counts.n_sanitisable else 0.0
                    ),
                    "novel": len(distinct - training) / len(distinct) if distinct else 0.0,
                    "generate_s": round(elapsed, 2),
                }
            )


class _Counts:
    """Per-draw tallies, kept apart from the rates computed from them."""

    __slots__ = ("distinct", "n_empty", "n_parseable", "n_sanitisable")

    def __init__(self) -> None:
        self.n_empty = 0
        self.n_parseable = 0
        self.n_sanitisable = 0
        self.distinct: set[str] = set()


def _classify(smiles: list[str]) -> _Counts:
    """Tally one draw under each definition of valid.

    Parsing and sanitisation are asked as two separate questions, which
    means ``sanitize=False`` on the way in: a single
    ``Chem.MolFromSmiles(s)`` collapses them, returning ``None`` whether
    the string was unreadable or merely chemically impossible, and those
    are different failures at different rates.

    An empty decode is counted as neither parseable nor sanitisable even
    though ``Chem.MolFromSmiles("")`` returns a molecule rather than
    ``None``. A zero-atom graph is not a molecule, and counting it as one
    inflates the rate.
    """
    counts = _Counts()
    for smi in smiles:
        if not smi.strip():
            counts.n_empty += 1
            continue
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        counts.n_parseable += 1
        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            continue
        counts.n_sanitisable += 1
        counts.distinct.add(Chem.MolToSmiles(mol))
    return counts


def summarise(rows: list[dict]) -> dict[float, dict[str, tuple[float, float]]]:
    """Mean and across-seed standard deviation of each rate, per temperature.

    Lives here rather than in a notebook so the numbers quoted in README.md
    are derived from the record by code that is itself reviewed, and so a
    reader can recompute them from ``rows.jsonl`` alone.
    """
    out: dict[float, dict[str, tuple[float, float]]] = {}
    for temperature in sorted({row["temperature"] for row in rows}):
        cell = [row for row in rows if row["temperature"] == temperature]
        out[temperature] = {
            rate: (
                float(np.mean([row[rate] for row in cell])),
                float(np.std([row[rate] for row in cell])),
            )
            for rate in ("parseable", "sanitisable", "unique", "novel")
        }
    return out

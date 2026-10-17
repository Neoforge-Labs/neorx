# GenMol — molecule generation

GenMol is a β-VAE over SMILES strings. Given a causal target, the
pipeline samples it for candidate molecules; on its own it is a small
generative model you can sample, interpolate in, or condition on
properties.

```
SMILES ──► Encoder (3-layer bidirectional GRU) ──► μ, log σ²
                                                      │
                                                 z ~ N(μ, σ²)
                                                      │
         Decoder (autoregressive GRU) ◄───────────────┘
                │
            SMILES
```

## The shipped checkpoint

A trained model ships inside the wheel, so generation works on a fresh
install with no download and no training step:

```python
>>> from neorx.genmol import load_pretrained
>>> model, tokenizer = load_pretrained()
>>> tokenizer.vocab_size
34
>>> sum(p.numel() for p in model.parameters())
4140322
>>> model.latent_dim
128

```

4.1 M parameters, a 128-dimensional latent space, and a 34-token
vocabulary learned from ChEMBL 36.

!!! note "The loader raises rather than degrading"
    `load_pretrained` never returns an untrained model. A randomly
    initialised VAE emits plausible-looking SMILES, so silent degradation
    here would be undetectable downstream — a missing or corrupt asset
    raises `GenMolAssetError` instead.

    ```python
    >>> from neorx.genmol import GenMolAssetError
    >>> issubclass(GenMolAssetError, RuntimeError)
    True

    ```

## Sampling

```python
>>> import torch
>>> from neorx.genmol import generate
>>> from neorx.molscreen import parse_smiles
>>> _ = torch.manual_seed(0)
>>> molecules = generate(model, tokenizer, n=20)
>>> 0 < len(molecules) <= 20
True
>>> all(parse_smiles(s) is not None for s in molecules)
True

```

`generate` validates with RDKit and deduplicates by default, so it
returns *at most* `n` molecules. Both are switchable:

```python
>>> _ = torch.manual_seed(0)
>>> raw = generate(model, tokenizer, n=20, validate=False, deduplicate=False)
>>> len(raw)
20

```

!!! warning "A measured validity rate, not a claimed one"
    Sampling 200 molecules at `torch.manual_seed(0)` on this machine
    (Python 3.13, macOS arm64, PyTorch 2.x) gave a validity of **0.855**
    and a uniqueness of **1.0** among the survivors.

    That figure is not asserted as a doctest, because it depends on the
    PyTorch build and the platform RNG, and a number that moves with the
    runner is not something a gate should enforce. A rate you can cite
    comes from a recorded run — see
    [Experiments and run records](experiments.md).

## Evaluating a sample

```python
>>> from neorx.genmol.evaluation import validity, uniqueness, internal_diversity
>>> validity(["CCO", "c1ccccc1", "not-a-molecule"])
0.6666666666666666
>>> uniqueness(["CCO", "CCO", "c1ccccc1"])
0.6666666666666666

```

`novelty` compares against a reference set, `internal_diversity`
averages pairwise Tanimoto distance, and `compute_all_metrics` runs the
set. `neorx.genmol.evaluation.distribution` adds property-distribution
comparison and a KS test against a reference library.

## Interpolating in latent space

A VAE's selling point is a smooth latent space: nearby points should
decode to similar molecules, which makes interpolation between two known
drugs meaningful.

```python
>>> from neorx.genmol import interpolate
>>> _ = torch.manual_seed(0)
>>> path = interpolate(model, tokenizer,
...                    "CC(=O)Oc1ccccc1C(O)=O",      # aspirin
...                    "CN1CCC[C@H]1c1cccnc1",       # nicotine
...                    n_steps=5)
>>> len(path)
5

```

The decoded path is not shown here, and that is deliberate: on this pair
with this checkpoint, all five steps decode to the *same* molecule. The
latent neighbourhood between these two points is flatter than the
interpolation story implies. Treat `interpolate` as a tool for inspecting
the latent space rather than as a source of novel candidates, and check
what it actually returns for your pair.

## Conditional generation

`MolCVAE` conditions on three scalar properties — molecular weight,
LogP and QED — by concatenating a property vector to both the encoder
input and the decoder's latent vector, so structure and properties are
disentangled and you can ask for a target profile at generation time.

```python
>>> from neorx.genmol import MolCVAE, PropertyNormalizer
>>> cvae = MolCVAE(vocab_size=tokenizer.vocab_size)
>>> cvae.latent_dim
128

```

No conditional checkpoint ships with the package; `MolCVAE` needs
training before it generates anything meaningful. Use
`generate_conditional` once you have one.

## Training your own

```python
>>> from neorx.genmol import TrainConfig
>>> config = TrainConfig()
>>> config.epochs, config.lr, config.grad_clip
(100, 0.001, 1.0)
>>> config.beta_start, config.beta_end, config.anneal_epochs
(0.01, 1.0, 10)

```

The KL annealing schedule is the part worth understanding. β on the KL
term rises linearly from 0.01 to 1.0 over the first ten epochs; without
annealing the model falls into posterior collapse, where the encoder
ignores its input and the decoder degenerates into an unconditional
language model that happens to emit valid SMILES.

From the command line:

```bash
neorx genmol download            # drug-like molecules from ChEMBL
neorx genmol train               # train the VAE
neorx genmol generate --n 100
neorx genmol evaluate
neorx genmol interpolate
neorx genmol screen              # pipe generated molecules through MolScreen
```

## Design notes

**GRU rather than LSTM** — fewer parameters and faster training; SMILES
sequences under 120 tokens do not need LSTM's extra gates.

**Teacher forcing** during training: the decoder receives the ground-truth
previous token rather than its own prediction, which stabilises early
training.

**Three-layer bidirectional encoder**, so the representation sees both
forward and backward context in the string.

## Next in the pipeline

Generated molecules go to [MolScreen](molscreen.md) for drug-likeness and
[DockBot](dockbot.md) for binding, then to the
[composite scorer](scoring.md).

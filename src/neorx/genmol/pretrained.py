"""Loading of the GenMol weights shipped with the package.

The loader raises on missing or inconsistent assets. It never returns an
untrained model: a randomly initialised VAE emits plausible-looking SMILES,
so silent degradation here is undetectable downstream.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .data.tokenizer import SmilesTokenizer
from .models.vae import MolVAE
from .train import load_checkpoint

ASSET_DIR = Path(__file__).parent / "assets"
CHECKPOINT_NAME = "molvae_chembl36.pt"
TOKENIZER_NAME = "tokenizer.json"


class GenMolAssetError(RuntimeError):
    """Raised when the shipped model assets are missing or unusable."""


def load_pretrained(
    device: torch.device | None = None,
) -> tuple[MolVAE, SmilesTokenizer]:
    """Load the packaged MolVAE and its matching tokenizer.

    Returns
    -------
    tuple[MolVAE, SmilesTokenizer]
        Model in eval mode, and the tokenizer it was trained against.

    Raises
    ------
    GenMolAssetError
        If either asset is absent, or the tokenizer carries no vocabulary.
    """
    ckpt_path = ASSET_DIR / CHECKPOINT_NAME
    tok_path = ASSET_DIR / TOKENIZER_NAME

    if not ckpt_path.exists() or not tok_path.exists():
        raise GenMolAssetError(
            f"no trained checkpoint found in {ASSET_DIR}. "
            "Reinstall neorx, or train one with `neorx genmol train`."
        )

    tokenizer = SmilesTokenizer.load(tok_path)
    if tokenizer.vocab_size == 0:
        raise GenMolAssetError(
            f"tokenizer at {tok_path} loaded but its vocabulary is empty. "
            "The asset is corrupt; reinstall neorx."
        )

    model = MolVAE(vocab_size=tokenizer.vocab_size)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    return model, tokenizer

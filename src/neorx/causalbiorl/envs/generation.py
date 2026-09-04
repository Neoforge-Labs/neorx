"""
Batched latent-space decoding.

The VAE decoder is a single forward pass, so decoding one molecule at a
time wastes almost all of it. Measured on CPU: 13.9 ms per molecule
one at a time, 2.0 ms per molecule at batch 200 -- a sevenfold
difference. A planner that scores candidate molecules needs the batched
form or it cannot afford to decode at all.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = ["decode_latent", "decode_latent_batch"]


def decode_latent(model, tokenizer, z: NDArray[np.floating]) -> str:
    """Decode a single latent vector to SMILES via the trained VAE.

    Uses greedy decoding rather than temperature sampling: the model
    has documented partial posterior collapse, and sampled decodes
    measurably produce chemically-invalid SMILES at the temperature=1.0
    default (verified empirically -- sampling yields ~85-95% validity
    over random latents, greedy yields ~100% in the same test). An
    invalid SMILES in a training step is a wasted step, not a useful
    exploration signal, so greedy is the right default here.
    """
    z_tensor = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        token_ids = model.decode(z_tensor, greedy=True)
    return tokenizer.decode(token_ids[0].tolist())


def decode_latent_batch(model, tokenizer, Z: NDArray[np.floating]) -> list[str]:
    """Decode a batch of latent vectors to SMILES.

    Greedy decoding, matching ``DrugDiscoveryEnv._decode_latent``: the
    shipped model has partial posterior collapse, and sampled decodes
    produce chemically invalid SMILES often enough to waste steps.

    Parameters
    ----------
    Z
        Array of shape ``(n, latent_dim)``.
    """
    if Z.ndim != 2:
        raise ValueError(
            f"decode_latent_batch expects a 2-D array of latents, got shape {Z.shape}"
        )
    if Z.shape[1] != model.latent_dim:
        raise ValueError(
            f"latent vectors have width {Z.shape[1]} but the decoder's latent "
            f"dimension is {model.latent_dim}"
        )

    z_tensor = torch.as_tensor(np.ascontiguousarray(Z), dtype=torch.float32)
    with torch.no_grad():
        token_ids = model.decode(z_tensor, greedy=True)

    return [tokenizer.decode(row.tolist()) for row in token_ids]

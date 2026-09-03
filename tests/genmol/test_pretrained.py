"""Shipped weights load, and their absence raises rather than degrading."""

import pytest

from neorx.genmol import GenMolAssetError, load_pretrained


def test_load_pretrained_returns_model_and_tokenizer():
    model, tok = load_pretrained()
    assert tok.vocab_size == 34
    assert not model.training, "model must be returned in eval mode"


def test_generated_molecules_are_valid():
    from rdkit import Chem

    from neorx.genmol import generate

    model, tok = load_pretrained()
    smiles = generate(model, tok, n=20)
    assert smiles, "generate returned nothing"
    assert all(Chem.MolFromSmiles(s) is not None for s in smiles)
    assert len(set(smiles)) > 1, "expected more than one distinct molecule"


def test_missing_asset_raises_not_degrades(monkeypatch, tmp_path):
    import neorx.genmol.pretrained as p

    monkeypatch.setattr(p, "ASSET_DIR", tmp_path)
    with pytest.raises(GenMolAssetError, match="no trained checkpoint"):
        p.load_pretrained()


def test_empty_vocabulary_raises(monkeypatch, tmp_path):
    import json

    import neorx.genmol.pretrained as p

    (tmp_path / "molvae_chembl36.pt").write_bytes(b"stub")
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"max_length": 120, "token_to_idx": {}})
    )
    monkeypatch.setattr(p, "ASSET_DIR", tmp_path)
    with pytest.raises(GenMolAssetError, match="vocabulary is empty"):
        p.load_pretrained()

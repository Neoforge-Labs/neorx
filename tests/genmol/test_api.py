"""The REST service generates from the shipped weights, or says why not.

These tests exist because the service used to load
``checkpoints/genmol/best_model.pt``: a path absent from every fresh
checkout, which nothing in the package creates, and which — where a stale
training run had left it behind — held different weights from the ones the
package ships.  A test that only asserts ``200`` would have passed in that
second case, so the assertions here are about *which* model is served.
"""

import inspect

import pytest
import torch
from fastapi.testclient import TestClient

from neorx.genmol import api, load_pretrained


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Clear the lazily loaded model so each test exercises a real load."""
    api._model = api._tokenizer = api._device = None
    yield
    api._model = api._tokenizer = api._device = None


@pytest.fixture
def client():
    return TestClient(api.app)


def test_generate_serves_the_packaged_weights(client):
    """Parameters served must be the shipped checkpoint's, to the bit."""
    assert client.post("/generate", json={"n": 4}).status_code == 200

    reference, _ = load_pretrained()
    served = api._model.state_dict()
    expected = reference.state_dict()

    assert set(served) == set(expected)
    mismatched = [
        name
        for name, tensor in expected.items()
        if not torch.equal(served[name].cpu(), tensor.cpu())
    ]
    assert not mismatched, (
        "served parameters differ from the packaged checkpoint: "
        f"{mismatched[:5]}"
    )


def test_generate_uses_the_matching_shipped_tokenizer(client):
    """A model is only as good as the vocabulary it was trained against."""
    assert client.post("/generate", json={"n": 4}).status_code == 200

    _, expected = load_pretrained()
    assert api._tokenizer.vocab_size == expected.vocab_size == 34
    assert api._tokenizer.token_to_idx == expected.token_to_idx


def test_generate_returns_parsable_molecules(client):
    from rdkit import Chem

    response = client.post("/generate", json={"n": 20, "temperature": 0.8})
    assert response.status_code == 200
    smiles = response.json()["smiles"]
    assert smiles, "endpoint returned no molecules"
    assert all(Chem.MolFromSmiles(s) is not None for s in smiles)


def test_missing_assets_give_an_actionable_503(client, monkeypatch, tmp_path):
    """The cause and the remedy belong in the response, not just the log."""
    import neorx.genmol.pretrained as pretrained

    monkeypatch.setattr(pretrained, "ASSET_DIR", tmp_path)

    response = client.post("/generate", json={"n": 4})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "no trained checkpoint" in detail
    assert "Reinstall neorx" in detail, (
        "the 503 must tell an operator what to do about it"
    )


def test_interpolate_also_reports_missing_assets(
    client, monkeypatch, tmp_path
):
    import neorx.genmol.pretrained as pretrained

    monkeypatch.setattr(pretrained, "ASSET_DIR", tmp_path)

    response = client.post(
        "/interpolate", json={"smiles_a": "CCO", "smiles_b": "c1ccccc1"}
    )
    assert response.status_code == 503
    assert "Reinstall neorx" in response.json()["detail"]


def test_load_model_accepts_no_checkpoint_override():
    """No path parameter: a second source of weights is the defect itself."""
    assert inspect.signature(api._load_model).parameters == {}


def test_unexpected_load_failure_is_not_relabelled_as_missing_assets(
    monkeypatch,
):
    """A bug in the loader must not masquerade as an absent checkpoint."""

    def boom(*args, **kwargs):
        raise RuntimeError("something else entirely")

    monkeypatch.setattr(api, "load_pretrained", boom)

    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.post("/generate", json={"n": 4})
    assert response.status_code == 500
    assert "Reinstall neorx" not in response.text

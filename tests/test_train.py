"""
Tests for train script helpers.
"""

import pytest

from scripts import train


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)

    assert train.resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    assert train.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cpu(monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)

    assert train.resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    assert train.resolve_device("cuda") == "cpu"


def test_resolve_device_rejects_unknown_device():
    with pytest.raises(ValueError):
        train.resolve_device("tpu")

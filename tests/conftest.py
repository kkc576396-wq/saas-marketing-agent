"""Suite-wide isolation for persistent memory and external embedding calls."""

import pytest


@pytest.fixture(autouse=True)
def disable_default_memory_side_effects(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    monkeypatch.setenv("MEMORY_EMBEDDING_ENABLED", "false")

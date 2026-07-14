"""Suppress noisy RDKit stderr during routine reaction / sanitize calls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rdkit import RDLogger


@contextmanager
def suppress_rdkit_logs() -> Iterator[None]:
    """Mute rdApp warnings/errors while GenMolRL handles failures in Python.

    Under ``masking=r2_available`` many (state, template, R2) tuples pass SMARTS
    pattern checks but ``RunReactants`` still emits kekulization / valence
    messages before we return ``None``. Those are expected, not actionable.
    """
    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")

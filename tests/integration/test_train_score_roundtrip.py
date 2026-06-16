"""Integration: CLI train -> score round-trip producing a loadable booster JSON.

Filled in against the real implementations: ``train`` (on the synthetic panel)
emits a ``<2MB`` booster JSON + fitted pipeline + manifest, ``score`` loads them
and scores a single application, and the round-trip is asserted end-to-end.

While the orchestration is a stub, this asserts the entry points exist and the
training/CLI surface raises the documented ``NotImplementedError`` until filled.
"""

from __future__ import annotations

import pytest

from lendingclub_default.cli import build_app
from lendingclub_default.train import train


@pytest.mark.integration
def test_train_entry_point_exists() -> None:
    """The end-to-end ``train`` entry point exists (raises until implemented)."""
    with pytest.raises(NotImplementedError):
        train(data_path=None)


@pytest.mark.integration
def test_cli_builder_entry_point_exists() -> None:
    """The Typer app builder exists (raises until the CLI is implemented)."""
    with pytest.raises(NotImplementedError):
        build_app()

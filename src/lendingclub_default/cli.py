"""Command-line interface (Typer).

A thin orchestration layer over the compute library: train the model (on a real
Kaggle CSV or the synthetic generator), score a single loan application, or
evaluate a held-out panel. Typer is built on the standard library, but
constructing the app object is deferred to :func:`build_app` so importing this
module has no side effects (no command registration or I/O at import time). The
module-level ``app`` is a lazily-built singleton consumed by the
``lendingclub-default`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer

# quantcore-candidate: new code (Typer CLI); lazy Typer import.


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the ``train``, ``score``, and ``evaluate`` commands on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    that importing :mod:`lendingclub_default.cli` does not import Typer or
    register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the CLI author.
    """
    raise NotImplementedError("build_app is not yet implemented.")


def app() -> None:
    """Console-script entry point: build the Typer app and invoke it.

    Wraps :func:`build_app` so ``lendingclub-default ...`` on the command line
    runs the CLI. Kept as a function (not a module-level Typer instance) to
    preserve import purity.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the CLI author.
    """
    raise NotImplementedError("app is not yet implemented.")

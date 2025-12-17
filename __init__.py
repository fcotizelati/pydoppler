"""Development-time import shim for local checkouts.

When this repository is checked out into a directory named ``pydoppler`` and the
parent directory is on ``sys.path`` (for example when starting Python from
``~``), ``import pydoppler`` would otherwise resolve to the repository root as an
implicit namespace package. That namespace has no public API, leading to errors
like ``AttributeError: module 'pydoppler' has no attribute 'spruit'``.

This file turns the repository root into a regular package and re-exports the
real implementation package living in ``pydoppler/`` so the README examples work
without having to ``cd`` into the checkout or install the project first.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any

_impl = _import_module(__name__ + ".pydoppler")

__all__ = list(getattr(_impl, "__all__", ()))

for _name in __all__:
    globals()[_name] = getattr(_impl, _name)

__version__ = getattr(_impl, "__version__", "0.0.0")


def __getattr__(name: str) -> _Any:  # pragma: no cover
    return getattr(_impl, name)


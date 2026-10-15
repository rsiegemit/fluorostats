"""Keeping renamed keyword arguments and moved names working for one deprecation cycle."""

from __future__ import annotations

import functools
import warnings
from types import ModuleType
from typing import Any, Callable


def renamed_keyword(old: str, new: str) -> Callable:
    """Decorator: accept keyword ``old`` as ``new``, with a DeprecationWarning.

    Passing both names is a TypeError. Positional calls are unaffected.
    """

    def wrap(func: Callable) -> Callable:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{func.__name__}() got both {old!r} (deprecated) and {new!r}")
                warnings.warn(
                    f"{func.__name__}(): {old!r} is deprecated; use {new!r}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return inner

    return wrap


def moved_names(module: str, names: tuple[str, ...], target: ModuleType) -> Callable[[str], Any]:
    """A module-level ``__getattr__`` for ``module``: each of ``names`` resolves to
    the same name in ``target``, with a DeprecationWarning; anything else is the
    usual AttributeError."""

    def __getattr__(name: str) -> Any:
        if name not in names:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        short = module.rsplit(".", 1)[-1]
        warnings.warn(
            f"{short}.{name} is deprecated; use {target.__name__}.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(target, name)

    return __getattr__


__all__ = ["renamed_keyword", "moved_names"]

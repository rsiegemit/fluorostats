"""Keeping renamed keyword arguments working for one deprecation cycle."""

from __future__ import annotations

import functools
import warnings
from typing import Callable


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


__all__ = ["renamed_keyword"]

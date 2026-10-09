# SPDX-License-Identifier: Apache-2.0
"""Spawn-safe bootstrap for host-created Python child processes."""

from __future__ import annotations

from typing import Any


def child_entry(target: Any, *args: Any, **kwargs: Any) -> Any:
    """Register inside a fresh interpreter, then run the host target."""

    from .plugin import register

    register()
    return target(*args, **kwargs)


__all__ = ["child_entry"]

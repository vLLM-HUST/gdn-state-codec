# SPDX-License-Identifier: Apache-2.0
"""Default-off ECPA activation entry point for the GDN state codec."""

from __future__ import annotations

import os
import sys

ENV = "GDN_STATE_CODEC"


def _var(name: str) -> str:
    return f"VLLM_HUST_{ENV}_{name}"


def _enabled(name: str) -> bool:
    return os.getenv(_var(name), "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def emit_evidence(message: str) -> None:
    print(f"EVIDENCE {message} pid={os.getpid()}", file=sys.stderr, flush=True)


_runtime_effective_emitted = False


def note_runtime_effective() -> None:
    """Report the first actual codec dispatch in this process."""

    global _runtime_effective_emitted
    if _runtime_effective_emitted or not _enabled("EVIDENCE"):
        return
    _runtime_effective_emitted = True
    emit_evidence("runtime_effective mechanism=gdn_state_codec")


def register() -> None:
    """Entry point called by vLLM in API, engine-core, and worker processes."""

    if _enabled("KILL_SWITCH"):
        return
    if not _enabled("ENABLE"):
        return
    from .adapter import install

    receipt = install()
    if _enabled("EVIDENCE"):
        emit_evidence(
            "installed mechanism=gdn_state_codec "
            f"host={receipt['host']} mode={receipt['mode']}"
        )


__all__ = ["register"]

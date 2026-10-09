# SPDX-License-Identifier: Apache-2.0
"""Fail-closed activation contract for the GDN state-codec experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_QWEN35_GENERATION_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


@dataclass(frozen=True, slots=True)
class GdnStateCodecExperiment:
    library_dir: Path
    mode: str
    initializer: str
    boundary_power_iterations: int
    decode_backend: str


def parse_gdn_state_codec_experiment(
    additional_config: dict[str, Any] | object,
    architectures: list[str] | tuple[str, ...],
    device_type: str,
    enforce_eager: bool,
    speculative_method: str | None,
    num_speculative_tokens: int,
) -> GdnStateCodecExperiment | None:
    """Validate the Manifest 0.3 configuration before loading native code."""

    if not isinstance(additional_config, dict):
        return None
    raw = additional_config.get("gdn_state_codec")
    if raw is None:
        return None
    expected = {
        "initializer",
        "boundary_power_iterations",
        "decode_backend",
        "mode",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(
            "gdn_state_codec must contain exactly boundary_power_iterations, "
            "decode_backend, initializer, and mode"
        )

    library_dir = os.environ.get("GDN_STATE_CODEC_LIBRARY_DIR", "")
    library_path = Path(library_dir)
    if not library_dir or not library_path.is_absolute():
        raise ValueError(
            "GDN_STATE_CODEC_LIBRARY_DIR must name an absolute native-library directory"
        )

    mode = raw["mode"]
    initializer = raw["initializer"]
    boundary_power_iterations = raw["boundary_power_iterations"]
    decode_backend = raw["decode_backend"]
    if mode not in {"shadow", "replace"}:
        raise ValueError("gdn_state_codec.mode must be 'shadow' or 'replace'")
    if initializer not in {"exact_svd", "quant_only"}:
        raise ValueError(
            "gdn_state_codec.initializer must be 'exact_svd' or 'quant_only'"
        )
    if decode_backend not in {"vector", "cube"}:
        raise ValueError("gdn_state_codec.decode_backend must be 'vector' or 'cube'")
    if (
        not isinstance(boundary_power_iterations, int)
        or isinstance(boundary_power_iterations, bool)
        or not 0 <= boundary_power_iterations <= 32
        or (boundary_power_iterations == 0 and initializer != "quant_only")
    ):
        raise ValueError(
            "gdn_state_codec.boundary_power_iterations must be in [1,32], "
            "or 0 with quant_only initialization"
        )
    if device_type != "npu":
        raise ValueError("gdn_state_codec only supports Ascend NPU")
    if not set(architectures).intersection(_QWEN35_GENERATION_ARCHITECTURES):
        raise ValueError("gdn_state_codec only supports Qwen3.5 generation models")
    if not enforce_eager:
        raise ValueError("gdn_state_codec currently requires enforce_eager")
    if speculative_method is not None or num_speculative_tokens != 0:
        raise ValueError("gdn_state_codec currently excludes speculative decoding")
    return GdnStateCodecExperiment(
        library_dir=library_path,
        mode=mode,
        initializer=initializer,
        boundary_power_iterations=boundary_power_iterations,
        decode_backend=decode_backend,
    )


__all__ = ["GdnStateCodecExperiment", "parse_gdn_state_codec_experiment"]

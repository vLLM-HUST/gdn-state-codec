# SPDX-License-Identifier: Apache-2.0
"""Version-bounded host adapter installed by the vLLM plugin entry point."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import sys
from types import ModuleType
from typing import Any

_MARKER = "_gdn_state_codec_extension_0_2_installed"
_TARGET_MODULE = "vllm_ascend.patch.worker.patch_qwen3_5"
_FINDER_MARKER = "_gdn_state_codec_deferred_finder"
_EXPECTED_HOST_MODULE = "vllm_ascend.ops.gdn"
_EXPECTED_SOURCE_MARKERS = (
    "npu_recurrent_gated_delta_rule",
    "non_spec_state_indices_tensor",
    "num_decodes",
)


def _guard_host(host_class: Any) -> None:
    original = getattr(host_class, "_forward_core", None)
    if original is None:
        raise RuntimeError("gdn_state_codec cannot find the host _forward_core")
    if original.__module__ != _EXPECTED_HOST_MODULE:
        raise RuntimeError(
            "gdn_state_codec refuses an unsupported or already patched GDN host: "
            f"{original.__module__}"
        )
    source = inspect.getsource(original)
    missing = [marker for marker in _EXPECTED_SOURCE_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            "gdn_state_codec host source guard failed; missing markers: "
            f"{missing}"
        )


def _install_now() -> dict[str, Any]:
    """Patch GDN only after vLLM-Ascend has installed its base adapter."""

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as host_class,
    )

    if getattr(host_class, _MARKER, False):
        return {"host": host_class.__module__, "mode": "already-installed"}

    _guard_host(host_class)

    from vllm.logger import init_logger

    from .host_gdn import GdnStateCodecForwardCore
    from .runtime import GdnStateCodecRuntime

    logger = init_logger(__name__)
    original_init = host_class.__init__

    def codec_init(
        self: Any,
        config: Any,
        vllm_config: Any,
        prefix: str = "",
        gqa_interleaved_layout: bool = False,
    ) -> None:
        original_init(self, config, vllm_config, prefix, gqa_interleaved_layout)
        runtime = GdnStateCodecRuntime.from_vllm_config(vllm_config, prefix)
        self._gdn_state_codec_runtime = runtime
        if runtime is not None:
            experiment = runtime.experiment
            logger.warning(
                "GDN State Codec active for %s in %s mode with %s initializer, "
                "%s decode backend, and %d boundary iterations",
                prefix,
                experiment.mode,
                experiment.initializer,
                experiment.decode_backend,
                experiment.boundary_power_iterations,
            )

    host_class.__init__ = codec_init
    host_class._forward_core = GdnStateCodecForwardCore._forward_core
    setattr(host_class, _MARKER, True)
    return {"host": host_class.__module__, "mode": "installed"}


class _AfterAscendLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _remove_deferred_finder()
        _install_now()


class _AfterAscendFinder(importlib.abc.MetaPathFinder):
    _gdn_state_codec_deferred_finder = True

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _AfterAscendLoader(spec.loader)
        return spec


def _remove_deferred_finder() -> None:
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not getattr(finder, _FINDER_MARKER, False)
    ]


def install() -> dict[str, Any]:
    """Install now or defer until Ascend's Qwen3.5 patch finishes importing."""

    if _TARGET_MODULE in sys.modules:
        return _install_now()
    if not any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AfterAscendFinder())
    return {"host": _TARGET_MODULE, "mode": "deferred"}


__all__ = ["install"]

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .config import GdnStateCodecExperiment, parse_gdn_state_codec_experiment

logger = init_logger(__name__)

KEY_DIMENSION = 128
VALUE_DIMENSION = 128
COMPENSATOR_RANK = 4
WINDOW_SIZE = 16
POSITION_STRIDE = 8
SHADOW_RELATIVE_RMS_LIMIT = 2.0e-2


@dataclass(slots=True)
class _EncodedSlot:
    residual: torch.Tensor
    smoothing: torch.Tensor
    scales: torch.Tensor
    compensator_keys: torch.Tensor
    compensator_values: torch.Tensor
    record_decay: torch.Tensor
    record_keys: torch.Tensor
    record_corrections: torch.Tensor
    positions: torch.Tensor
    scratch_residual: torch.Tensor
    scratch_smoothing: torch.Tensor
    scratch_scales: torch.Tensor
    scratch_compensator_keys: torch.Tensor
    scratch_compensator_values: torch.Tensor
    residual_workspace: torch.Tensor
    scaled_record_keys: torch.Tensor
    output: torch.Tensor
    kernel_state_pointers: tuple[int, ...] = ()
    steps: int = 0


class _GdnStateCodecLibraries:
    def __init__(self, library_dir: Path) -> None:
        # The runtime libraries have DT_NEEDED edges to kernels in the same
        # experiment directory.  Do not rely on a StateAxis-owned
        # LD_LIBRARY_PATH: an ECPA Bundle must be self-sufficient once the
        # Manager supplies its declared library directory.
        self._local_dependencies = []
        for dependency_name in (
            "libstatecentric_gdn_state_codec_window_boundary_kernel.so",
            "libstatecentric_gdn_state_codec_bf16_cube_matmul_kernel.so",
            "libstatecentric_gdn_state_codec_int8_cube_matmul_kernel.so",
            "libstatecentric_gdn_state_codec_encoded_projection_kernel.so",
            "libstatecentric_gdn_state_codec_smoothing_kernel.so",
            "libstatecentric_gdn_state_codec_quantize_kernel.so",
        ):
            dependency = library_dir / dependency_name
            if not dependency.is_file():
                raise RuntimeError(
                    f"GDN State Codec native dependency is missing: {dependency}"
                )
            self._local_dependencies.append(
                ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
            )
        library = (
            library_dir
            / "libstatecentric_gdn_state_codec_windowed_gdn_step_kernel.so"
        )
        if not library.is_file():
            raise RuntimeError(f"GDN State Codec kernel library is missing: {library}")
        self._window = ctypes.CDLL(str(library))
        self.window_step = (
            self._window.statecentric_gdn_state_codec_windowed_gdn_step_launch_v1
        )
        self.window_step.argtypes = [ctypes.c_void_p] * 16 + [ctypes.c_uint32]
        self.window_step.restype = ctypes.c_int32
        self.window_step_bf16 = getattr(
            self._window,
            "statecentric_gdn_state_codec_windowed_gdn_step_bf16_launch_v1",
            None,
        )
        if self.window_step_bf16 is not None:
            self.window_step_bf16.argtypes = (
                [ctypes.c_void_p] * 16 + [ctypes.c_uint32]
            )
            self.window_step_bf16.restype = ctypes.c_int32
        self.window_step_bf16_quant_only = getattr(
            self._window,
            "statecentric_gdn_state_codec_windowed_gdn_step_bf16_quant_only_launch_v1",
            None,
        )
        if self.window_step_bf16_quant_only is not None:
            self.window_step_bf16_quant_only.argtypes = (
                [ctypes.c_void_p] * 16 + [ctypes.c_uint32]
            )
            self.window_step_bf16_quant_only.restype = ctypes.c_int32
        boundary_library = (
            library_dir
            / "libstatecentric_gdn_state_codec_window_boundary_kernel.so"
        )
        if not boundary_library.is_file():
            raise RuntimeError(
                f"GDN State Codec boundary library is missing: {boundary_library}"
            )
        self._boundary = ctypes.CDLL(str(boundary_library))
        self.window_boundary = (
            self._boundary.statecentric_gdn_state_codec_window_boundary_launch_v1
        )
        self.window_boundary.argtypes = (
            [ctypes.c_void_p] * 16
            + [ctypes.c_uint32, ctypes.c_uint32]
        )
        self.window_boundary.restype = ctypes.c_int32
        self.window_boundary_cube_finish = getattr(
            self._boundary,
            "statecentric_gdn_state_codec_window_boundary_cube_finish_launch_v1",
            None,
        )
        if self.window_boundary_cube_finish is not None:
            self.window_boundary_cube_finish.argtypes = (
                [ctypes.c_void_p] * 16 + [ctypes.c_uint32]
            )
            self.window_boundary_cube_finish.restype = ctypes.c_int32
        self.boundary_runtime_step = None
        self._boundary_runtime_create = None
        self._boundary_runtime_destroy = None
        self._boundary_handle: int | None = None
        boundary_runtime_library = (
            library_dir / "libstatecentric_gdn_state_codec_boundary_runtime.so"
        )
        if boundary_runtime_library.is_file():
            self._boundary_runtime = ctypes.CDLL(
                str(boundary_runtime_library)
            )
            self._boundary_runtime_create = (
                self._boundary_runtime
                .statecentric_gdn_state_codec_boundary_runtime_create_v1
            )
            self._boundary_runtime_create.argtypes = [ctypes.c_uint32]
            self._boundary_runtime_create.restype = ctypes.c_void_p
            self._boundary_runtime_destroy = (
                self._boundary_runtime
                .statecentric_gdn_state_codec_boundary_runtime_destroy_v1
            )
            self._boundary_runtime_destroy.argtypes = [ctypes.c_void_p]
            self._boundary_runtime_destroy.restype = None
            self.boundary_runtime_step = (
                self._boundary_runtime
                .statecentric_gdn_state_codec_boundary_runtime_step_v1
            )
            self.boundary_runtime_step.argtypes = (
                [ctypes.c_void_p] * 7 + [ctypes.c_uint32]
            )
            self.boundary_runtime_step.restype = ctypes.c_int32
        decode_library = library_dir / "libstatecentric_gdn_state_codec_decode_runtime.so"
        self.decode_runtime_step = None
        self._decode_runtime_create = None
        self._decode_runtime_destroy = None
        self._decode_handle: int | None = None
        if decode_library.is_file():
            self._decode = ctypes.CDLL(str(decode_library))
            self._decode_runtime_create = (
                self._decode.statecentric_gdn_state_codec_decode_runtime_create_v1
            )
            self._decode_runtime_create.argtypes = [ctypes.c_uint32]
            self._decode_runtime_create.restype = ctypes.c_void_p
            self._decode_runtime_destroy = (
                self._decode.statecentric_gdn_state_codec_decode_runtime_destroy_v1
            )
            self._decode_runtime_destroy.argtypes = [ctypes.c_void_p]
            self._decode_runtime_destroy.restype = None
            self.decode_runtime_step = (
                self._decode.statecentric_gdn_state_codec_decode_runtime_step_v1
            )
            self.decode_runtime_step.argtypes = (
                [ctypes.c_void_p] * 17 + [ctypes.c_uint32]
            )
            self.decode_runtime_step.restype = ctypes.c_int32

        self.provider_smoothing = None
        smoothing_library = (
            library_dir / "libstatecentric_gdn_state_codec_smoothing_kernel.so"
        )
        if smoothing_library.is_file():
            self._smoothing = ctypes.CDLL(str(smoothing_library))
            self.provider_smoothing = getattr(
                self._smoothing,
                "statecentric_gdn_state_codec_provider_smoothing_launch_v2",
                None,
            )
            if self.provider_smoothing is not None:
                self.provider_smoothing.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                ]
                self.provider_smoothing.restype = ctypes.c_int32

        self.provider_quantize = None
        quantize_library = (
            library_dir / "libstatecentric_gdn_state_codec_quantize_kernel.so"
        )
        if quantize_library.is_file():
            self._quantize = ctypes.CDLL(str(quantize_library))
            self.provider_quantize = getattr(
                self._quantize,
                "statecentric_gdn_state_codec_provider_quantize_launch_v2",
                None,
            )
            if self.provider_quantize is not None:
                self.provider_quantize.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                ]
                self.provider_quantize.restype = ctypes.c_int32

        self.provider_initialize = None
        provider_runtime_library = (
            library_dir
            / "libstatecentric_gdn_state_codec_provider_runtime.so"
        )
        if provider_runtime_library.is_file():
            self._provider_runtime = ctypes.CDLL(
                str(provider_runtime_library)
            )
            self.provider_initialize = getattr(
                self._provider_runtime,
                "statecentric_gdn_state_codec_provider_initialize_launch_v3",
                None,
            )
            if self.provider_initialize is not None:
                self.provider_initialize.argtypes = (
                    [ctypes.c_void_p] * 5 + [ctypes.c_uint32]
                )
                self.provider_initialize.restype = ctypes.c_int32

    def decode_handle(self, states: int) -> int | None:
        if self._decode_runtime_create is None:
            return None
        if self._decode_handle is None:
            handle = self._decode_runtime_create(ctypes.c_uint32(states))
            if not handle:
                raise RuntimeError("GDN State Codec Cube decode runtime creation failed")
            self._decode_handle = int(handle)
        return self._decode_handle

    def boundary_handle(self, states: int) -> int | None:
        if self._boundary_runtime_create is None:
            return None
        if self._boundary_handle is None:
            handle = self._boundary_runtime_create(ctypes.c_uint32(states))
            if not handle:
                raise RuntimeError(
                    "GDN State Codec Cube boundary runtime creation failed"
                )
            self._boundary_handle = int(handle)
        return self._boundary_handle

    def close(self) -> None:
        if (
            self._decode_handle is not None
            and self._decode_runtime_destroy is not None
        ):
            self._decode_runtime_destroy(ctypes.c_void_p(self._decode_handle))
            self._decode_handle = None
        if (
            self._boundary_handle is not None
            and self._boundary_runtime_destroy is not None
        ):
            self._boundary_runtime_destroy(
                ctypes.c_void_p(self._boundary_handle)
            )
            self._boundary_handle = None


class GdnStateCodecRuntime:
    """Eager, single-request runtime gate for Qwen3.5 GDN decode.

    This first integration stage deliberately uses device-side exact SVD only
    when a prefill-produced dense state enters the compressed representation.
    The steady-state token step is the native p=16, r=4 AscendC kernel.
    """

    def __init__(
        self, experiment: GdnStateCodecExperiment, prefix: str
    ) -> None:
        self.experiment = experiment
        self.prefix = prefix
        self._libraries = _GdnStateCodecLibraries(experiment.library_dir)
        self._slots: dict[int, _EncodedSlot] = {}
        # The experiment is admitted only under enforce_eager, and a vLLM
        # model worker executes it on one stream.  Resolve that stream lazily
        # on the first token and avoid a ~25 us Python current_stream query on
        # every decode step.
        self._execution_stream: torch.npu.Stream | None = None
        self._input_contract_validated = False
        self.shadow_steps = 0
        self.shadow_max_relative_rms = 0.0
        self.boundary_refits = 0
        self.native_fallbacks = 0
        self.native_fallback_reasons: dict[str, int] = {}

    def __del__(self) -> None:
        libraries = getattr(self, "_libraries", None)
        if libraries is not None:
            libraries.close()

    @staticmethod
    def from_vllm_config(
        vllm_config: VllmConfig, prefix: str
    ) -> GdnStateCodecRuntime | None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        architectures = list(getattr(hf_config, "architectures", ()) or ())
        speculative_config = vllm_config.speculative_config
        speculative_method = (
            getattr(speculative_config, "method", None)
            if speculative_config is not None
            else None
        )
        speculative_tokens = (
            int(getattr(speculative_config, "num_speculative_tokens", 0))
            if speculative_config is not None
            else 0
        )
        experiment = parse_gdn_state_codec_experiment(
            vllm_config.additional_config,
            architectures,
            vllm_config.device_config.device_type,
            model_config.enforce_eager,
            speculative_method,
            speculative_tokens,
        )
        if experiment is None:
            return None
        return GdnStateCodecRuntime(experiment, prefix)

    @staticmethod
    def _encode_dense_boundary(
        dense: torch.Tensor, initializer: str = "exact_svd"
    ) -> _EncodedSlot:
        if dense.ndim != 3 or dense.shape[-2:] != (
            KEY_DIMENSION,
            VALUE_DIMENSION,
        ):
            raise RuntimeError(
                "GDN State Codec requires per-head [128,128] recurrent states"
            )
        dense = dense.to(torch.float32).contiguous()
        if initializer == "exact_svd":
            left_vectors, singular_values, right_vectors_t = torch.linalg.svd(
                dense, full_matrices=False
            )
            left = (
                left_vectors[:, :, :COMPENSATOR_RANK]
                * singular_values[:, None, :COMPENSATOR_RANK]
            )
            right = right_vectors_t[:, :COMPENSATOR_RANK, :].transpose(-2, -1)
            residual = dense - torch.bmm(left, right.transpose(-2, -1))
        elif initializer == "quant_only":
            left = torch.zeros(
                (dense.shape[0], KEY_DIMENSION, COMPENSATOR_RANK),
                dtype=torch.float32,
                device=dense.device,
            )
            right = torch.zeros(
                (dense.shape[0], VALUE_DIMENSION, COMPENSATOR_RANK),
                dtype=torch.float32,
                device=dense.device,
            )
            residual = dense
        else:
            raise RuntimeError(
                f"unsupported GdnStateCodec initializer: {initializer}"
            )
        smoothing = torch.sqrt(
            torch.clamp(residual.abs().mean(dim=2), min=1.0e-8)
        )
        smoothed = residual / smoothing.unsqueeze(2)
        scales = smoothed.abs().amax(dim=1)
        safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        quantized = torch.round(
            smoothed / safe_scales.unsqueeze(1) * 127.0
        ).clamp_(-127, 127)
        encoded_residual = (
            quantized.to(torch.int8).transpose(-2, -1).contiguous()
        )
        encoded_smoothing = smoothing.contiguous()
        encoded_scales = scales.contiguous()
        encoded_compensator_keys = left.to(torch.float16).contiguous()
        encoded_compensator_values = right.to(torch.float16).contiguous()
        return GdnStateCodecRuntime._allocate_slot(
            encoded_residual,
            encoded_smoothing,
            encoded_scales,
            encoded_compensator_keys,
            encoded_compensator_values,
        )

    @staticmethod
    def _allocate_slot(
        encoded_residual: torch.Tensor,
        encoded_smoothing: torch.Tensor,
        encoded_scales: torch.Tensor,
        encoded_compensator_keys: torch.Tensor,
        encoded_compensator_values: torch.Tensor,
    ) -> _EncodedSlot:
        heads = encoded_residual.shape[0]
        device = encoded_residual.device
        record_keys = torch.zeros(
            (heads, WINDOW_SIZE, KEY_DIMENSION),
            dtype=torch.float16,
            device=device,
        )
        slot = _EncodedSlot(
            # Store [value,key] so a decode read can consume one contiguous
            # 128-byte INT8 row per output value in the vector kernel.
            residual=encoded_residual,
            smoothing=encoded_smoothing,
            scales=encoded_scales,
            compensator_keys=encoded_compensator_keys,
            compensator_values=encoded_compensator_values,
            record_decay=torch.ones(
                (heads, WINDOW_SIZE), dtype=torch.float32, device=device
            ),
            record_keys=record_keys,
            record_corrections=torch.zeros(
                (heads, WINDOW_SIZE, VALUE_DIMENSION),
                dtype=torch.float16,
                device=device,
            ),
            # Ascend PyTorch does not implement zero_ for uint32. The native
            # kernel interprets these non-negative int32 bits as uint32.
            positions=torch.zeros(
                (heads, POSITION_STRIDE),
                dtype=torch.int32,
                device=device,
            ),
            scratch_residual=torch.empty_like(encoded_residual),
            scratch_smoothing=torch.empty_like(encoded_smoothing),
            scratch_scales=torch.empty_like(encoded_scales),
            scratch_compensator_keys=torch.empty_like(
                encoded_compensator_keys
            ),
            scratch_compensator_values=torch.empty_like(
                encoded_compensator_values
            ),
            residual_workspace=torch.empty(
                encoded_residual.shape,
                dtype=torch.float32,
                device=device,
            ),
            scaled_record_keys=torch.empty_like(record_keys),
            output=torch.empty(
                (1, 1, heads, VALUE_DIMENSION),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        slot.kernel_state_pointers = (
            slot.residual.data_ptr(),
            slot.smoothing.data_ptr(),
            slot.scales.data_ptr(),
            slot.compensator_keys.data_ptr(),
            slot.compensator_values.data_ptr(),
            slot.record_decay.data_ptr(),
            slot.record_keys.data_ptr(),
            slot.record_corrections.data_ptr(),
            slot.positions.data_ptr(),
        )
        return slot

    def _encode_quant_only_provider(
        self, provider_dense: torch.Tensor
    ) -> _EncodedSlot:
        if provider_dense.ndim != 3 or provider_dense.shape[-2:] != (
            VALUE_DIMENSION,
            KEY_DIMENSION,
        ):
            raise RuntimeError(
                "GDN State Codec requires provider states shaped [H,128,128]"
            )
        if provider_dense.dtype != torch.float32:
            raise RuntimeError("GDN State Codec provider state must be FP32")
        if not provider_dense.is_contiguous():
            raise RuntimeError("GDN State Codec provider state must be contiguous")
        initialize_launch = self._libraries.provider_initialize
        smoothing_launch = self._libraries.provider_smoothing
        quantize_launch = self._libraries.provider_quantize
        if initialize_launch is None and (
            smoothing_launch is None or quantize_launch is None
        ):
            raise RuntimeError(
                "GDN State Codec batched provider initializer is unavailable"
            )

        heads = provider_dense.shape[0]
        residual = torch.empty_like(provider_dense, dtype=torch.int8)
        smoothing = torch.empty(
            (heads, KEY_DIMENSION),
            dtype=torch.float32,
            device=provider_dense.device,
        )
        scales = torch.empty(
            (heads, VALUE_DIMENSION),
            dtype=torch.float32,
            device=provider_dense.device,
        )
        compensator_keys = torch.zeros(
            (heads, KEY_DIMENSION, COMPENSATOR_RANK),
            dtype=torch.float16,
            device=provider_dense.device,
        )
        compensator_values = torch.zeros(
            (heads, VALUE_DIMENSION, COMPENSATOR_RANK),
            dtype=torch.float16,
            device=provider_dense.device,
        )
        stream = self._stream()
        if initialize_launch is not None:
            status = initialize_launch(
                stream.npu_stream,
                provider_dense.data_ptr(),
                smoothing.data_ptr(),
                residual.data_ptr(),
                scales.data_ptr(),
                heads,
            )
            if status != 0:
                raise RuntimeError(
                    "GDN State Codec provider initializer launch failed: "
                    f"{status}"
                )
        else:
            status = smoothing_launch(
                stream.npu_stream,
                provider_dense.data_ptr(),
                smoothing.data_ptr(),
                heads,
            )
            if status != 0:
                raise RuntimeError(
                    f"GDN State Codec provider smoothing launch failed: {status}"
                )
            status = quantize_launch(
                stream.npu_stream,
                provider_dense.data_ptr(),
                smoothing.data_ptr(),
                residual.data_ptr(),
                scales.data_ptr(),
                heads,
            )
            if status != 0:
                raise RuntimeError(
                    f"GDN State Codec provider quantize launch failed: {status}"
                )
        return self._allocate_slot(
            residual,
            smoothing,
            scales,
            compensator_keys,
            compensator_values,
        )

    @staticmethod
    def reconstruct(slot: _EncodedSlot) -> torch.Tensor:
        residual = (
            slot.residual.transpose(-2, -1).to(torch.float32)
            / 127.0
            * slot.smoothing.unsqueeze(2)
            * slot.scales.unsqueeze(1)
        )
        compensation = torch.bmm(
            slot.compensator_keys.to(torch.float32),
            slot.compensator_values.to(torch.float32).transpose(-2, -1),
        )
        return residual + compensation

    def invalidate(self, cache_indices: tuple[int, ...]) -> None:
        for cache_index in cache_indices:
            self._slots.pop(cache_index, None)

    @staticmethod
    def decode_fallback_reason(
        *,
        num_prefills: int,
        num_decodes: int,
        speculative: bool,
        cache_index_count: int | None = None,
    ) -> str | None:
        """Return why this invocation must use the native recurrent op."""
        if speculative:
            return "speculative decode"
        if num_prefills > 0 and num_decodes > 0:
            return "mixed prefill/decode batch"
        if num_decodes > 0 and (
            num_decodes != 1
            or (cache_index_count is not None and cache_index_count != 1)
        ):
            return "multi-request decode batch"
        return None

    def fall_back_to_native(self, reason: str) -> None:
        """Discard compressed shadows before the provider updates dense state."""
        cleared_slots = len(self._slots)
        self._slots.clear()
        self.native_fallbacks += 1
        reason_count = self.native_fallback_reasons.get(reason, 0) + 1
        self.native_fallback_reasons[reason] = reason_count
        if reason_count == 1:
            logger.info(
                "GDN State Codec %s uses native fallback for %s "
                "(%d compressed slots invalidated)",
                self.prefix,
                reason,
                cleared_slots,
            )

    def _slot(self, provider_state: torch.Tensor, cache_index: int) -> _EncodedSlot:
        slot = self._slots.get(cache_index)
        if slot is not None:
            return slot
        if cache_index < 0 or cache_index >= provider_state.shape[0]:
            raise RuntimeError("GDN State Codec cache index is out of range")
        provider_dense = provider_state[cache_index]
        if self.experiment.initializer == "quant_only":
            slot = self._encode_quant_only_provider(provider_dense)
        else:
            dense = provider_dense.transpose(-2, -1).contiguous()
            slot = self._encode_dense_boundary(
                dense, self.experiment.initializer
            )
        if (
            self.experiment.boundary_power_iterations == 0
            and self._libraries.boundary_runtime_step is not None
            and self._libraries.window_boundary_cube_finish is not None
        ):
            self._libraries.boundary_handle(slot.residual.shape[0])
        self._slots[cache_index] = slot
        logger.info("GDN State Codec initialized %s cache slot %d", self.prefix, cache_index)
        return slot

    def _refit_boundary(self, slot: _EncodedSlot) -> None:
        next_residual = slot.scratch_residual
        next_smoothing = slot.scratch_smoothing
        next_scales = slot.scratch_scales
        next_compensator_keys = slot.scratch_compensator_keys
        next_compensator_values = slot.scratch_compensator_values
        residual_workspace = slot.residual_workspace
        stream = self._stream()
        use_cube_boundary = (
            self.experiment.boundary_power_iterations == 0
            and self._libraries.boundary_runtime_step is not None
            and self._libraries.window_boundary_cube_finish is not None
        )
        if use_cube_boundary:
            scaled_record_keys = slot.scaled_record_keys
            boundary_handle = self._libraries.boundary_handle(
                slot.residual.shape[0]
            )
            if boundary_handle is None:
                raise RuntimeError(
                    "GDN State Codec Cube boundary runtime is unavailable"
                )
            merge_status = self._libraries.boundary_runtime_step(
                boundary_handle,
                stream.npu_stream,
                slot.record_decay.data_ptr(),
                slot.record_keys.data_ptr(),
                slot.record_corrections.data_ptr(),
                scaled_record_keys.data_ptr(),
                residual_workspace.data_ptr(),
                slot.residual.shape[0],
            )
            if merge_status != 0:
                raise RuntimeError(
                    f"GDN State Codec Cube boundary merge failed: {merge_status}"
                )
        pointers = (
            stream.npu_stream,
            slot.residual.data_ptr(),
            slot.smoothing.data_ptr(),
            slot.scales.data_ptr(),
            slot.compensator_keys.data_ptr(),
            slot.compensator_values.data_ptr(),
            slot.record_decay.data_ptr(),
            slot.record_keys.data_ptr(),
            slot.record_corrections.data_ptr(),
            slot.positions.data_ptr(),
            next_residual.data_ptr(),
            next_smoothing.data_ptr(),
            next_scales.data_ptr(),
            next_compensator_keys.data_ptr(),
            next_compensator_values.data_ptr(),
            residual_workspace.data_ptr(),
        )
        if use_cube_boundary:
            status = self._libraries.window_boundary_cube_finish(
                *pointers,
                slot.residual.shape[0],
            )
        else:
            status = self._libraries.window_boundary(
                *pointers,
                slot.residual.shape[0],
                self.experiment.boundary_power_iterations,
            )
        if status != 0:
            raise RuntimeError(
                f"GDN State Codec boundary-refit launch failed: {status}"
            )
        # The launch bypasses PyTorch's dispatcher. Record the external stream
        # on buffers that lose their Python owner below so the caching
        # allocator cannot recycle them before the asynchronous refit reads
        # complete. The next step is enqueued on the same stream.
        slot.residual.record_stream(stream)
        slot.smoothing.record_stream(stream)
        slot.scales.record_stream(stream)
        slot.compensator_keys.record_stream(stream)
        slot.compensator_values.record_stream(stream)
        residual_workspace.record_stream(stream)
        if use_cube_boundary:
            scaled_record_keys.record_stream(stream)
        slot.residual, slot.scratch_residual = next_residual, slot.residual
        slot.smoothing, slot.scratch_smoothing = (
            next_smoothing,
            slot.smoothing,
        )
        slot.scales, slot.scratch_scales = next_scales, slot.scales
        slot.compensator_keys, slot.scratch_compensator_keys = (
            next_compensator_keys,
            slot.compensator_keys,
        )
        slot.compensator_values, slot.scratch_compensator_values = (
            next_compensator_values,
            slot.compensator_values,
        )
        slot.kernel_state_pointers = (
            slot.residual.data_ptr(),
            slot.smoothing.data_ptr(),
            slot.scales.data_ptr(),
            slot.compensator_keys.data_ptr(),
            slot.compensator_values.data_ptr(),
            slot.record_decay.data_ptr(),
            slot.record_keys.data_ptr(),
            slot.record_corrections.data_ptr(),
            slot.positions.data_ptr(),
        )
        slot.steps = 0
        self.boundary_refits += 1
        log_boundary = (
            logger.warning
            if self.experiment.mode == "shadow"
            else logger.info
        )
        log_boundary(
            "GDN State Codec boundary %s refits=%d shadow_max_relative_rms=%.8g",
            self.prefix,
            self.boundary_refits,
            self.shadow_max_relative_rms,
        )

    def _stream(self) -> torch.npu.Stream:
        stream = self._execution_stream
        if stream is None:
            stream = torch.npu.current_stream()
            self._execution_stream = stream
        return stream

    def step(
        self,
        provider_state: torch.Tensor,
        cache_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        query_scale: float,
    ) -> torch.Tensor:
        slot = self._slot(provider_state, cache_index)
        if slot.steps == WINDOW_SIZE:
            self._refit_boundary(slot)
        elif slot.steps > WINDOW_SIZE:
            raise RuntimeError(
                "GDN State Codec runtime window position exceeded p=16"
            )
        heads = slot.residual.shape[0]
        if not self._input_contract_validated:
            query_shape = (1, 1, heads // 2, KEY_DIMENSION)
            value_shape = (1, 1, heads, VALUE_DIMENSION)
            scalar_shape = (1, 1, heads)
            if query.shape != query_shape or key.shape != query_shape:
                raise RuntimeError(
                    "GDN State Codec Qwen3.5 gate requires [1,1,H/2,128] query/key"
                )
            if value.shape != value_shape:
                raise RuntimeError(
                    "GDN State Codec Qwen3.5 gate requires [1,1,H,128] value"
                )
            if log_decay.shape != scalar_shape or beta.shape != scalar_shape:
                raise RuntimeError(
                    "GDN State Codec Qwen3.5 gate requires [1,1,H] decay/beta"
                )
            if log_decay.dtype != torch.float32:
                raise RuntimeError(
                    "GDN State Codec Qwen3.5 gate requires FP32 log decay"
                )
            if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
                raise RuntimeError(
                    "GDN State Codec Qwen3.5 gate requires BF16 query/key"
                )
            if value.dtype != torch.bfloat16 or beta.dtype != torch.bfloat16:
                raise RuntimeError(
                    "GDN State Codec Qwen3.5 gate requires BF16 value/beta"
                )
            if abs(query_scale - KEY_DIMENSION**-0.5) > 1.0e-8:
                raise RuntimeError("GDN State Codec Qwen3.5 query scale is unsupported")
            if not all(
                tensor.is_contiguous()
                for tensor in (query, key, value, log_decay, beta)
            ):
                raise RuntimeError("GDN State Codec Qwen3.5 inputs must be contiguous")
            self._input_contract_validated = True
        outputs = slot.output
        stream = self._stream().npu_stream
        pointers = (
            stream,
            *slot.kernel_state_pointers,
            log_decay.data_ptr(),
            beta.data_ptr(),
            key.data_ptr(),
            query.data_ptr(),
            value.data_ptr(),
            outputs.data_ptr(),
        )
        if self.experiment.decode_backend == "vector":
            vector_step = self._libraries.window_step_bf16
            if (
                self.experiment.initializer == "quant_only"
                and self.experiment.boundary_power_iterations == 0
                and self._libraries.window_step_bf16_quant_only is not None
            ):
                vector_step = self._libraries.window_step_bf16_quant_only
            if vector_step is None:
                raise RuntimeError(
                    "GDN State Codec vector decode backend is unavailable"
                )
            status = vector_step(
                *pointers,
                heads,
            )
        else:
            decode_handle = self._libraries.decode_handle(heads)
            if decode_handle is None or self._libraries.decode_runtime_step is None:
                raise RuntimeError("GDN State Codec Cube decode backend is unavailable")
            status = self._libraries.decode_runtime_step(
                decode_handle,
                *pointers,
                heads,
            )
        if status != 0:
            raise RuntimeError(f"GDN State Codec decode-step launch failed: {status}")
        slot.steps += 1
        return outputs

    def observe_shadow(
        self, compressed: torch.Tensor, reference: torch.Tensor
    ) -> float:
        difference = compressed.to(torch.float32) - reference.to(torch.float32)
        denominator = reference.to(torch.float32).square().sum().clamp_min(1.0e-30)
        relative_rms = torch.sqrt(difference.square().sum() / denominator)
        relative_rms_value = float(relative_rms.cpu())
        self.shadow_steps += 1
        self.shadow_max_relative_rms = max(
            self.shadow_max_relative_rms, relative_rms_value
        )
        logger.info(
            "GDN State Codec shadow %s step=%d relative_rms=%.8g max=%.8g",
            self.prefix,
            self.shadow_steps,
            relative_rms_value,
            self.shadow_max_relative_rms,
        )
        if relative_rms_value >= SHADOW_RELATIVE_RMS_LIMIT:
            raise RuntimeError(
                "GDN State Codec shadow output exceeded the relative-RMS gate: "
                f"{relative_rms_value:.8g}"
            )
        return relative_rms_value

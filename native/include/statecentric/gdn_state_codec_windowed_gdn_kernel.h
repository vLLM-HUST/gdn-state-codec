#pragma once

#include <cstdint>

// One state is one GDN head with dk=dv=128. Boundary tensors use [state,key,value],
// compensators use [state,axis,4], record tensors use [state,16,axis], and
// positions use [state,8] so independently written values occupy cache lines.
extern "C" std::int32_t statecentric_gdn_state_codec_windowed_gdn_step_launch_v1(
    void* stream, const std::int8_t* residual, const float* smoothing,
    const float* scales, const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, const float* decay,
    const std::uint16_t* write_keys, const std::uint16_t* reads,
    const std::uint16_t* queries, const std::uint16_t* values, float* outputs,
    std::uint32_t states);

// Qwen3.5-35B decode specialization. Q/K have half as many heads as V and are
// repeated twice by the kernel. All model-facing tensors are native BF16.
extern "C" std::int32_t
statecentric_gdn_state_codec_windowed_gdn_step_bf16_launch_v1(
    void* stream, const std::int8_t* residual, const float* smoothing,
    const float* scales, const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, const float* decay, const std::uint16_t* beta,
    const std::uint16_t* keys, const std::uint16_t* queries,
    const std::uint16_t* values, std::uint16_t* outputs,
    std::uint32_t states);

extern "C" std::int32_t
statecentric_gdn_state_codec_windowed_gdn_step_bf16_quant_only_launch_v1(
    void* stream, const std::int8_t* residual, const float* smoothing,
    const float* scales, const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, const float* decay, const std::uint16_t* beta,
    const std::uint16_t* keys, const std::uint16_t* queries,
    const std::uint16_t* values, std::uint16_t* outputs,
    std::uint32_t states);

extern "C" std::int32_t statecentric_gdn_state_codec_fp32_gdn_step_launch_v1(
    void* stream, float* dense_state, const float* decay,
    const std::uint16_t* write_keys, const std::uint16_t* reads,
    const std::uint16_t* queries, const std::uint16_t* values, float* outputs,
    std::uint32_t states);

// Refit a full p=16 window into a new INT8 boundary plus r=4 FP16
// compensators. Input and output boundary buffers must not alias.
extern "C" std::int32_t statecentric_gdn_state_codec_window_boundary_launch_v1(
    void* stream, const std::int8_t* residual, const float* smoothing,
    const float* scales, const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, std::int8_t* next_residual,
    float* next_smoothing, float* next_scales,
    std::uint16_t* next_compensator_keys,
    std::uint16_t* next_compensator_values, float* residual_workspace,
    std::uint32_t states,
    std::uint32_t power_iterations);

extern "C" std::int32_t
statecentric_gdn_state_codec_boundary_scale_records_launch_v1(
    void* stream, const float* record_decay,
    const std::uint16_t* record_keys,
    std::uint16_t* scaled_record_keys,
    std::uint32_t states);

extern "C" std::int32_t
statecentric_gdn_state_codec_window_boundary_cube_finish_launch_v1(
    void* stream, const std::int8_t* residual, const float* smoothing,
    const float* scales, const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, std::int8_t* next_residual,
    float* next_smoothing, float* next_scales,
    std::uint16_t* next_compensator_keys,
    std::uint16_t* next_compensator_values, float* record_merge,
    std::uint32_t states);

extern "C" std::int32_t statecentric_gdn_state_codec_bf16_cube_matmul_launch_v1(
    void* stream, const std::uint16_t* a, const std::uint16_t* b, float* c,
    std::uint32_t states, void* workspace, void* tiling,
    std::uint32_t blocks);

// Synthetic Cube ABI probe. This is not an end-to-end GdnStateCodec result.
extern "C" std::int32_t statecentric_gdn_state_codec_cube_matmul_probe_launch_v1(
    void* stream, const std::uint16_t* a, const std::uint16_t* b, float* c,
    std::uint32_t states, std::uint32_t transpose_a, void* workspace,
    void* tiling, std::uint32_t blocks);

extern "C" std::int32_t statecentric_gdn_state_codec_cube_normalize_launch_v1(
    void* stream, const float* input, std::uint16_t* output,
    std::uint32_t states, std::uint32_t active_columns);

extern "C" std::int32_t
statecentric_gdn_state_codec_cube_deflate_normalize_launch_v1(
    void* stream, float* projected, const std::uint16_t* vector,
    const std::uint16_t* left_basis, const std::uint16_t* right_basis,
    std::uint16_t* output, std::uint32_t states,
    std::uint32_t completed_ranks, std::uint32_t transpose);

extern "C" std::int32_t statecentric_gdn_state_codec_cube_commit_rank_launch_v1(
    void* stream, float* projected, const std::uint16_t* right_vector,
    std::uint16_t* left_basis, std::uint16_t* right_basis,
    std::uint32_t states, std::uint32_t rank);

extern "C" std::int32_t statecentric_gdn_state_codec_int8_cube_matmul_launch_v1(
    void* stream, const std::int8_t* a, const std::int8_t* b,
    std::int32_t* c, std::uint32_t states, std::uint32_t transpose_a,
    void* workspace, void* tiling, std::uint32_t blocks);

extern "C" std::int32_t statecentric_gdn_state_codec_prepare_int8_vector_launch_v1(
    void* stream, const std::uint16_t* input, const float* axis_scale,
    std::int8_t* quantized, float* dynamic_scale, std::uint32_t states);

extern "C" std::int32_t statecentric_gdn_state_codec_prepare_decode_pair_launch_v1(
    void* stream, const std::uint16_t* reads, const std::uint16_t* queries,
    const float* decay, const float* smoothing, std::int8_t* quantized,
    float* dynamic_scale, std::uint32_t states);

extern "C" std::int32_t statecentric_gdn_state_codec_finish_decode_pair_launch_v1(
    void* stream, const std::int32_t* projected,
    const float* dynamic_scale, const float* scales,
    const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, const float* decay,
    const std::uint16_t* write_keys, const std::uint16_t* reads,
    const std::uint16_t* queries, const std::uint16_t* values,
    float* outputs, std::uint32_t states);

extern "C" std::int32_t
statecentric_gdn_state_codec_prepare_decode_pair_bf16_launch_v1(
    void* stream, const std::uint16_t* keys, const std::uint16_t* queries,
    const float* decay, const float* smoothing, std::int8_t* quantized,
    float* dynamic_scale, std::uint32_t states);

extern "C" std::int32_t
statecentric_gdn_state_codec_finish_decode_pair_bf16_launch_v1(
    void* stream, const std::int32_t* projected,
    const float* dynamic_scale, const float* scales,
    const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, const float* decay,
    const std::uint16_t* beta, const std::uint16_t* keys,
    const std::uint16_t* queries, const std::uint16_t* values,
    std::uint16_t* outputs, std::uint32_t states);

extern "C" std::int32_t
statecentric_gdn_state_codec_finish_encoded_projection_launch_v1(
    void* stream, const std::int32_t* projected, const float* dynamic_scale,
    const std::uint16_t* input, const float* smoothing, const float* scales,
    const std::uint16_t* old_left, const std::uint16_t* old_right,
    const float* record_decay, const std::uint16_t* record_left,
    const std::uint16_t* record_right, const std::uint16_t* new_left,
    const std::uint16_t* new_right, std::uint16_t* output,
    std::uint32_t states, std::uint32_t fitted_ranks,
    std::uint32_t transpose);

extern "C" std::int32_t statecentric_gdn_state_codec_commit_encoded_rank_launch_v1(
    void* stream, const std::int32_t* projected, const float* dynamic_scale,
    const std::uint16_t* input, const float* smoothing, const float* scales,
    const std::uint16_t* old_left, const std::uint16_t* old_right,
    const float* record_decay, const std::uint16_t* record_left,
    const std::uint16_t* record_right, std::uint16_t* new_left,
    std::uint16_t* new_right, float* corrected, std::uint32_t states,
    std::uint32_t rank);

extern "C" std::int32_t statecentric_gdn_state_codec_encoded_requantize_launch_v1(
    void* stream, const std::int8_t* residual, const float* smoothing,
    const float* scales, const std::uint16_t* old_left,
    const std::uint16_t* old_right, const float* record_decay,
    const std::uint16_t* record_left, const std::uint16_t* record_right,
    const std::uint16_t* new_left, const std::uint16_t* new_right,
    std::int8_t* next_residual, float* next_smoothing, float* next_scales,
    std::uint16_t* next_left, std::uint16_t* next_right,
    float* residual_workspace, std::uint32_t states);

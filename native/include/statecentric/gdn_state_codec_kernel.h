#pragma once

#include <cstdint>

extern "C" std::int32_t statecentric_gdn_state_codec_smoothing_launch_v1(
    void* stream, const float* residual, float* smoothing);

extern "C" std::int32_t statecentric_gdn_state_codec_quantize_launch_v1(
    void* stream, const float* residual, const float* smoothing,
    std::int8_t* quantized, float* value_scales);

extern "C" std::int32_t statecentric_gdn_state_codec_dequantize_launch_v1(
    void* stream, const std::int8_t* quantized, const float* smoothing,
    const float* value_scales, const float* compensator_keys,
    const float* compensator_values, float* dense);

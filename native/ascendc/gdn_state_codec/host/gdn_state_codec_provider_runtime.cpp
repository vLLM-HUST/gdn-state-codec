#include "statecentric/gdn_state_codec_kernel.h"

extern "C" std::int32_t
statecentric_gdn_state_codec_provider_initialize_launch_v3(
    void* stream, const float* provider_state, float* smoothing,
    std::int8_t* quantized, float* value_scales, std::uint32_t states) {
  const std::int32_t smoothing_status =
      statecentric_gdn_state_codec_provider_smoothing_launch_v2(
          stream, provider_state, smoothing, states);
  if (smoothing_status != 0) {
    return smoothing_status;
  }
  return statecentric_gdn_state_codec_provider_quantize_launch_v2(
      stream, provider_state, smoothing, quantized, value_scales, states);
}

#include "kernel_operator.h"

#include "statecentric/leapquant_windowed_gdn_kernel.h"

namespace {
constexpr uint32_t kRows = 128;
constexpr uint32_t kColumns = 16;

__aicore__ inline float SquareRoot(float value) {
  float estimate = value > 1.0F ? value : 1.0F;
  for (uint32_t iteration = 0; iteration < 20; ++iteration) {
    estimate = 0.5F * (estimate + value / estimate);
  }
  return estimate;
}
}  // namespace

extern "C" __global__ __aicore__ void statecentric_leapquant_cube_normalize(
    GM_ADDR input, GM_ADDR output, uint32_t states, uint32_t active_columns) {
  AscendC::GlobalTensor<float> input_global;
  AscendC::GlobalTensor<half> output_global;
  input_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input),
                               states * kRows * kColumns);
  output_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output),
                                states * kRows * kColumns);
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    const uint32_t base = state * kRows * kColumns;
    for (uint32_t column = 0; column < kColumns; ++column) {
      float squared_norm = 0.0F;
      if (column < active_columns) {
        for (uint32_t row = 0; row < kRows; ++row) {
          const float value =
              input_global.GetValue(base + row * kColumns + column);
          squared_norm += value * value;
        }
      }
      const float inverse_norm = squared_norm > 1.0e-20F
                                     ? 1.0F / SquareRoot(squared_norm)
                                     : 0.0F;
      for (uint32_t row = 0; row < kRows; ++row) {
        const float value = column < active_columns
                                ? input_global.GetValue(
                                      base + row * kColumns + column) *
                                      inverse_norm
                                : 0.0F;
        output_global.SetValue(base + row * kColumns + column,
                               static_cast<half>(value));
      }
    }
  }
}

extern "C" int32_t statecentric_leapquant_cube_normalize_launch_v1(
    void* stream, const float* input, uint16_t* output, uint32_t states,
    uint32_t active_columns) {
  if (!stream || !input || !output || states == 0 || states > 16384 ||
      active_columns == 0 || active_columns > kColumns) {
    return 1;
  }
  statecentric_leapquant_cube_normalize<<<64, nullptr, stream>>>(
      const_cast<float*>(input), reinterpret_cast<half*>(output), states,
      active_columns);
  return 0;
}

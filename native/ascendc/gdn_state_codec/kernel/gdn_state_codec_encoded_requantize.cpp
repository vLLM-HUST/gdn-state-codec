#include "kernel_operator.h"

#include "statecentric/gdn_state_codec_windowed_gdn_kernel.h"

namespace {
constexpr uint32_t kD = 128;
constexpr uint32_t kElements = kD * kD;
constexpr uint32_t kRank = 4;
constexpr uint32_t kWindow = 16;
constexpr float kFloor = 1.0e-8F;

__aicore__ inline float Absolute(float value) {
  return value < 0.0F ? -value : value;
}
__aicore__ inline float SquareRoot(float value) {
  float estimate = value > 1.0F ? value : 1.0F;
  for (uint32_t iteration = 0; iteration < 20; ++iteration) {
    estimate = 0.5F * (estimate + value / estimate);
  }
  return estimate;
}

class EncodedRequantize {
 public:
  __aicore__ inline void Init(
      GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales, GM_ADDR old_left,
      GM_ADDR old_right, GM_ADDR record_decay, GM_ADDR record_left,
      GM_ADDR record_right, GM_ADDR new_left, GM_ADDR new_right,
      GM_ADDR next_residual, GM_ADDR next_smoothing, GM_ADDR next_scales,
      GM_ADDR next_left, GM_ADDR next_right, GM_ADDR residual_workspace,
      uint32_t states) {
    states_ = states;
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(residual),
                              states * kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                               states * kD);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                            states * kD);
    old_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(old_left),
                              states * kD * kRank);
    old_right_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(old_right),
                               states * kD * kRank);
    record_decay_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(record_decay), states * kWindow);
    record_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_left),
                                 states * kWindow * kD);
    record_right_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_right),
                                  states * kWindow * kD);
    new_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(new_left),
                              states * kRank * kD);
    new_right_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(new_right),
                               states * kRank * kD);
    next_residual_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(next_residual), states * kElements);
    next_smoothing_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(next_smoothing), states * kD);
    next_scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(next_scales),
                                 states * kD);
    next_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(next_left),
                               states * kD * kRank);
    next_right_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(next_right),
                                states * kD * kRank);
    residual_workspace_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(residual_workspace),
        states * kElements);
  }

  __aicore__ inline float Cell(uint32_t state, uint32_t row, uint32_t column,
                               const float* record_suffix,
                               float boundary_suffix) {
    const uint32_t matrix_base = state * kElements;
    const uint32_t old_base = state * kD * kRank;
    const uint32_t new_base = state * kRank * kD;
    float value = static_cast<float>(residual_.GetValue(
                      matrix_base + row * kD + column)) /
                  127.0F * smoothing_.GetValue(state * kD + row) *
                  scales_.GetValue(state * kD + column) * boundary_suffix;
    for (uint32_t rank = 0; rank < kRank; ++rank) {
      value += static_cast<float>(
                   old_left_.GetValue(old_base + row * kRank + rank)) *
               static_cast<float>(
                   old_right_.GetValue(old_base + column * kRank + rank)) *
               boundary_suffix;
      value -= static_cast<float>(
                   new_left_.GetValue(new_base + rank * kD + row)) *
               static_cast<float>(
                   new_right_.GetValue(new_base + rank * kD + column));
    }
    for (uint32_t record = 0; record < kWindow; ++record) {
      const uint32_t record_base = (state * kWindow + record) * kD;
      value += static_cast<float>(record_left_.GetValue(record_base + row)) *
               static_cast<float>(
                   record_right_.GetValue(record_base + column)) *
               record_suffix[record];
    }
    return value;
  }

  __aicore__ inline void Process() {
    for (uint32_t state = AscendC::GetBlockIdx(); state < states_;
         state += AscendC::GetBlockNum()) {
      float record_suffix[kWindow];
      float suffix = 1.0F;
      for (uint32_t reverse = kWindow; reverse > 0; --reverse) {
        const uint32_t record = reverse - 1;
        record_suffix[record] = suffix;
        suffix *= record_decay_.GetValue(state * kWindow + record);
      }
      for (uint32_t row = 0; row < kD; ++row) {
        float mean_absolute = 0.0F;
        for (uint32_t column = 0; column < kD; ++column) {
          const float value =
              Cell(state, row, column, record_suffix, suffix);
          residual_workspace_.SetValue(
              matrix_base(state) + row * kD + column, value);
          mean_absolute += Absolute(value);
        }
        const float mean = mean_absolute / 128.0F;
        next_smoothing_.SetValue(state * kD + row,
                                 SquareRoot(mean > kFloor ? mean : kFloor));
      }
      for (uint32_t column = 0; column < kD; ++column) {
        float maximum = 0.0F;
        for (uint32_t row = 0; row < kD; ++row) {
          const float value = Absolute(
              residual_workspace_.GetValue(
                  matrix_base(state) + row * kD + column) /
              next_smoothing_.GetValue(state * kD + row));
          maximum = value > maximum ? value : maximum;
        }
        next_scales_.SetValue(state * kD + column, maximum);
        for (uint32_t row = 0; row < kD; ++row) {
          const float smoothed =
              residual_workspace_.GetValue(
                  matrix_base(state) + row * kD + column) /
              next_smoothing_.GetValue(state * kD + row);
          float quantized =
              maximum > 0.0F ? smoothed / maximum * 127.0F : 0.0F;
          quantized += quantized >= 0.0F ? 0.5F : -0.5F;
          quantized = quantized > 127.0F ? 127.0F : quantized;
          quantized = quantized < -127.0F ? -127.0F : quantized;
          next_residual_.SetValue(matrix_base(state) + row * kD + column,
                                  static_cast<int8_t>(quantized));
        }
      }
      const uint32_t output_base = state * kD * kRank;
      const uint32_t new_base = state * kRank * kD;
      for (uint32_t axis = 0; axis < kD; ++axis) {
        for (uint32_t rank = 0; rank < kRank; ++rank) {
          next_left_.SetValue(output_base + axis * kRank + rank,
                              new_left_.GetValue(new_base + rank * kD + axis));
          next_right_.SetValue(
              output_base + axis * kRank + rank,
              new_right_.GetValue(new_base + rank * kD + axis));
        }
      }
    }
  }

 private:
  __aicore__ inline uint32_t matrix_base(uint32_t state) const {
    return state * kElements;
  }
  uint32_t states_ = 0;
  AscendC::GlobalTensor<int8_t> residual_, next_residual_;
  AscendC::GlobalTensor<float> smoothing_, scales_, record_decay_,
      next_smoothing_, next_scales_, residual_workspace_;
  AscendC::GlobalTensor<half> old_left_, old_right_, record_left_,
      record_right_, new_left_, new_right_, next_left_, next_right_;
};
}  // namespace

extern "C" __global__ __aicore__ void
statecentric_gdn_state_codec_encoded_requantize(
    GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales, GM_ADDR old_left,
    GM_ADDR old_right, GM_ADDR record_decay, GM_ADDR record_left,
    GM_ADDR record_right, GM_ADDR new_left, GM_ADDR new_right,
    GM_ADDR next_residual, GM_ADDR next_smoothing, GM_ADDR next_scales,
    GM_ADDR next_left, GM_ADDR next_right, GM_ADDR residual_workspace,
    uint32_t states) {
  EncodedRequantize kernel;
  kernel.Init(residual, smoothing, scales, old_left, old_right, record_decay,
              record_left, record_right, new_left, new_right, next_residual,
              next_smoothing, next_scales, next_left, next_right,
              residual_workspace, states);
  kernel.Process();
}

extern "C" int32_t statecentric_gdn_state_codec_encoded_requantize_launch_v1(
    void* stream, const int8_t* residual, const float* smoothing,
    const float* scales, const uint16_t* old_left, const uint16_t* old_right,
    const float* record_decay, const uint16_t* record_left,
    const uint16_t* record_right, const uint16_t* new_left,
    const uint16_t* new_right, int8_t* next_residual, float* next_smoothing,
    float* next_scales, uint16_t* next_left, uint16_t* next_right,
    float* residual_workspace, uint32_t states) {
  if (!stream || !residual || !smoothing || !scales || !old_left ||
      !old_right || !record_decay || !record_left || !record_right ||
      !new_left || !new_right || !next_residual || !next_smoothing ||
      !next_scales || !next_left || !next_right || states == 0 ||
      !residual_workspace || states > 16384) return 1;
  statecentric_gdn_state_codec_encoded_requantize<<<64, nullptr, stream>>>(
      const_cast<int8_t*>(residual), const_cast<float*>(smoothing),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(old_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(old_right)),
      const_cast<float*>(record_decay),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_right)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(new_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(new_right)), next_residual,
      next_smoothing, next_scales, reinterpret_cast<half*>(next_left),
      reinterpret_cast<half*>(next_right), residual_workspace, states);
  return 0;
}

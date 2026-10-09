#include "kernel_operator.h"

#include "statecentric/gdn_state_codec_windowed_gdn_kernel.h"

namespace {
constexpr uint32_t kRows = 128;
constexpr uint32_t kColumns = 16;
constexpr uint32_t kRanks = 4;

__aicore__ inline float SquareRoot(float value) {
  float estimate = value > 1.0F ? value : 1.0F;
  for (uint32_t iteration = 0; iteration < 20; ++iteration) {
    estimate = 0.5F * (estimate + value / estimate);
  }
  return estimate;
}

__aicore__ inline float CorrectedValue(
    AscendC::GlobalTensor<float>& projected,
    AscendC::GlobalTensor<half>& left_basis,
    AscendC::GlobalTensor<half>& right_basis, uint32_t state,
    uint32_t row, const float* coefficients, uint32_t completed_ranks,
    uint32_t transpose) {
  const uint32_t projected_base = state * kRows * kColumns;
  const uint32_t basis_base = state * kRanks * kRows;
  float value = projected.GetValue(projected_base + row * kColumns);
  for (uint32_t rank = 0; rank < completed_ranks; ++rank) {
    const uint32_t index = basis_base + rank * kRows + row;
    const float output_basis = transpose == 0U
                                   ? static_cast<float>(left_basis.GetValue(index))
                                   : static_cast<float>(right_basis.GetValue(index));
    value -= output_basis * coefficients[rank];
  }
  return value;
}
}  // namespace

// Correct the dense Cube result for already extracted rank-1 factors, then
// normalize the active column. This keeps the original matrix immutable and
// matches sequential deflation without materializing a new dense residual.
extern "C" __global__ __aicore__ void
statecentric_gdn_state_codec_cube_deflate_normalize(
    GM_ADDR projected, GM_ADDR vector, GM_ADDR left_basis,
    GM_ADDR right_basis, GM_ADDR output, uint32_t states,
    uint32_t completed_ranks, uint32_t transpose) {
  AscendC::GlobalTensor<float> projected_global;
  AscendC::GlobalTensor<half> vector_global;
  AscendC::GlobalTensor<half> left_global;
  AscendC::GlobalTensor<half> right_global;
  AscendC::GlobalTensor<half> output_global;
  projected_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected),
                                   states * kRows * kColumns);
  vector_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vector),
                                states * kRows * kColumns);
  left_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(left_basis),
                              states * kRanks * kRows);
  right_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(right_basis),
                               states * kRanks * kRows);
  output_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output),
                                states * kRows * kColumns);

  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    float coefficients[kRanks] = {0.0F, 0.0F, 0.0F, 0.0F};
    const uint32_t projected_base = state * kRows * kColumns;
    const uint32_t basis_base = state * kRanks * kRows;
    for (uint32_t rank = 0; rank < completed_ranks; ++rank) {
      for (uint32_t inner = 0; inner < kRows; ++inner) {
        const uint32_t index = basis_base + rank * kRows + inner;
        const float basis_value =
            transpose == 0U
                ? static_cast<float>(right_global.GetValue(index))
                : static_cast<float>(left_global.GetValue(index));
        coefficients[rank] +=
            basis_value * static_cast<float>(vector_global.GetValue(
                              projected_base + inner * kColumns));
      }
    }
    float squared_norm = 0.0F;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float value = CorrectedValue(
          projected_global, left_global, right_global, state, row,
          coefficients, completed_ranks, transpose);
      squared_norm += value * value;
    }
    const float inverse_norm = squared_norm > 1.0e-20F
                                   ? 1.0F / SquareRoot(squared_norm)
                                   : 0.0F;
    const uint32_t output_base = projected_base;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float value = CorrectedValue(
          projected_global, left_global, right_global, state, row,
          coefficients, completed_ranks, transpose);
      output_global.SetValue(output_base + row * kColumns,
                             static_cast<half>(value * inverse_norm));
      for (uint32_t column = 1; column < kColumns; ++column) {
        output_global.SetValue(output_base + row * kColumns + column,
                               static_cast<half>(0.0F));
      }
    }
  }
}

// Apply the final forward deflation and commit the FP16 left/right factors.
// The corrected FP32 projection remains available for the correctness gate.
extern "C" __global__ __aicore__ void statecentric_gdn_state_codec_cube_commit_rank(
    GM_ADDR projected, GM_ADDR right_vector, GM_ADDR left_basis,
    GM_ADDR right_basis, uint32_t states, uint32_t rank) {
  AscendC::GlobalTensor<float> projected_global;
  AscendC::GlobalTensor<half> vector_global;
  AscendC::GlobalTensor<half> left_global;
  AscendC::GlobalTensor<half> right_global;
  projected_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected),
                                   states * kRows * kColumns);
  vector_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(right_vector),
                                states * kRows * kColumns);
  left_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(left_basis),
                              states * kRanks * kRows);
  right_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(right_basis),
                               states * kRanks * kRows);

  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    const uint32_t projected_base = state * kRows * kColumns;
    const uint32_t basis_base = state * kRanks * kRows + rank * kRows;
    const uint32_t state_basis_base = state * kRanks * kRows;
    float coefficients[kRanks] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (uint32_t prior_rank = 0; prior_rank < rank; ++prior_rank) {
      for (uint32_t inner = 0; inner < kRows; ++inner) {
        coefficients[prior_rank] +=
            static_cast<float>(right_global.GetValue(
                state_basis_base + prior_rank * kRows + inner)) *
            static_cast<float>(vector_global.GetValue(
                projected_base + inner * kColumns));
      }
    }
    for (uint32_t row = 0; row < kRows; ++row) {
      const float value = CorrectedValue(
          projected_global, left_global, right_global, state, row,
          coefficients, rank, 0U);
      projected_global.SetValue(projected_base + row * kColumns, value);
      left_global.SetValue(basis_base + row, static_cast<half>(value));
      right_global.SetValue(
          basis_base + row,
          vector_global.GetValue(projected_base + row * kColumns));
    }
  }
}

extern "C" int32_t statecentric_gdn_state_codec_cube_deflate_normalize_launch_v1(
    void* stream, float* projected, const uint16_t* vector,
    const uint16_t* left_basis, const uint16_t* right_basis,
    uint16_t* output, uint32_t states, uint32_t completed_ranks,
    uint32_t transpose) {
  if (!stream || !projected || !vector || !left_basis || !right_basis ||
      !output || states == 0 || states > 16384 ||
      completed_ranks > kRanks || transpose > 1U) {
    return 1;
  }
  statecentric_gdn_state_codec_cube_deflate_normalize<<<64, nullptr, stream>>>(
      projected, reinterpret_cast<half*>(const_cast<uint16_t*>(vector)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(left_basis)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(right_basis)),
      reinterpret_cast<half*>(output), states, completed_ranks, transpose);
  return 0;
}

extern "C" int32_t statecentric_gdn_state_codec_cube_commit_rank_launch_v1(
    void* stream, float* projected, const uint16_t* right_vector,
    uint16_t* left_basis, uint16_t* right_basis, uint32_t states,
    uint32_t rank) {
  if (!stream || !projected || !right_vector || !left_basis || !right_basis ||
      states == 0 || states > 16384 || rank >= kRanks) {
    return 1;
  }
  statecentric_gdn_state_codec_cube_commit_rank<<<64, nullptr, stream>>>(
      projected,
      reinterpret_cast<half*>(const_cast<uint16_t*>(right_vector)),
      reinterpret_cast<half*>(left_basis), reinterpret_cast<half*>(right_basis),
      states, rank);
  return 0;
}

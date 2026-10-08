#include "kernel_operator.h"

#include "statecentric/leapquant_windowed_gdn_kernel.h"

namespace {
constexpr uint32_t kRows = 128;
constexpr uint32_t kColumns = 16;
constexpr uint32_t kRanks = 4;
constexpr uint32_t kWindow = 16;
constexpr uint32_t kScalarStride = 8;

__aicore__ inline float Absolute(float value) {
  return value < 0.0F ? -value : value;
}

union FloatBits {
  uint32_t bits;
  float value;
};

__aicore__ inline float BFloat16ToFloat(uint16_t value) {
  FloatBits converted;
  converted.bits = static_cast<uint32_t>(value) << 16;
  return converted.value;
}

__aicore__ inline uint16_t FloatToBFloat16(float value) {
  FloatBits converted;
  converted.value = value;
  const uint32_t least_significant = (converted.bits >> 16) & 1U;
  return static_cast<uint16_t>(
      (converted.bits + 0x7FFFU + least_significant) >> 16);
}

__aicore__ inline float SquareRoot(float value) {
  float estimate = value > 1.0F ? value : 1.0F;
  for (uint32_t iteration = 0; iteration < 20; ++iteration) {
    estimate = 0.5F * (estimate + value / estimate);
  }
  return estimate;
}

class EncodedProjection {
 public:
  __aicore__ inline void Init(
      GM_ADDR projected, GM_ADDR dynamic_scale, GM_ADDR input,
      GM_ADDR smoothing, GM_ADDR scales, GM_ADDR old_left, GM_ADDR old_right,
      GM_ADDR record_decay, GM_ADDR record_left, GM_ADDR record_right,
      GM_ADDR new_left, GM_ADDR new_right, uint32_t states) {
    states_ = states;
    projected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(projected),
                               states * kRows * kColumns);
    dynamic_scale_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(dynamic_scale), states * kScalarStride);
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input),
                           states * kRows * kColumns);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                               states * kRows);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                            states * kRows);
    old_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(old_left),
                              states * kRows * kRanks);
    old_right_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(old_right),
                               states * kRows * kRanks);
    record_decay_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(record_decay), states * kWindow);
    record_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_left),
                                 states * kWindow * kRows);
    record_right_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(record_right),
        states * kWindow * kRows);
    new_left_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(new_left),
                              states * kRanks * kRows);
    new_right_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(new_right),
                               states * kRanks * kRows);
  }

  __aicore__ inline float PrepareState(uint32_t state, uint32_t fitted_ranks,
                                       uint32_t transpose,
                                       float* coefficients,
                                       float* record_coefficients) {
    const uint32_t vector_base = state * kRows * kColumns;
    const uint32_t old_base = state * kRows * kRanks;
    const uint32_t new_base = state * kRanks * kRows;
    for (uint32_t rank = 0; rank < kRanks; ++rank) coefficients[rank] = 0.0F;
    for (uint32_t record = 0; record < kWindow; ++record) {
      record_coefficients[record] = 0.0F;
    }
    for (uint32_t axis = 0; axis < kRows; ++axis) {
      const float input =
          static_cast<float>(input_.GetValue(vector_base + axis * kColumns));
      for (uint32_t rank = 0; rank < kRanks; ++rank) {
        const uint32_t old_index = old_base + axis * kRanks + rank;
        const uint32_t new_index = new_base + rank * kRows + axis;
        const float old_basis =
            transpose == 0U
                ? static_cast<float>(old_right_.GetValue(old_index))
                : static_cast<float>(old_left_.GetValue(old_index));
        coefficients[rank] += old_basis * input;
        if (rank < fitted_ranks) {
          const float new_basis =
              transpose == 0U
                  ? static_cast<float>(new_right_.GetValue(new_index))
                  : static_cast<float>(new_left_.GetValue(new_index));
          coefficients[kRanks + rank] += new_basis * input;
        }
      }
      for (uint32_t record = 0; record < kWindow; ++record) {
        const uint32_t index = (state * kWindow + record) * kRows + axis;
        const float basis =
            transpose == 0U
                ? static_cast<float>(record_right_.GetValue(index))
                : static_cast<float>(record_left_.GetValue(index));
        record_coefficients[record] += basis * input;
      }
    }
    float suffix = 1.0F;
    for (uint32_t reverse = kWindow; reverse > 0; --reverse) {
      const uint32_t record = reverse - 1;
      record_coefficients[record] *= suffix;
      suffix *= record_decay_.GetValue(state * kWindow + record);
    }
    return suffix;
  }

  __aicore__ inline float Value(uint32_t state, uint32_t row,
                                uint32_t fitted_ranks, uint32_t transpose,
                                const float* coefficients,
                                const float* record_coefficients,
                                float boundary_suffix) {
    const uint32_t vector_base = state * kRows;
    const uint32_t projected_base = state * kRows * kColumns;
    const uint32_t old_base = state * kRows * kRanks;
    const uint32_t new_base = state * kRanks * kRows;
    float value = static_cast<float>(
                      projected_.GetValue(projected_base + row * kColumns)) *
                  dynamic_scale_.GetValue(state * kScalarStride) / 127.0F *
                  (transpose == 0U ? smoothing_.GetValue(vector_base + row)
                                   : scales_.GetValue(vector_base + row)) *
                  boundary_suffix;
    for (uint32_t rank = 0; rank < kRanks; ++rank) {
      const uint32_t old_index = old_base + row * kRanks + rank;
      const float output_basis =
          transpose == 0U
              ? static_cast<float>(old_left_.GetValue(old_index))
              : static_cast<float>(old_right_.GetValue(old_index));
      value += output_basis * coefficients[rank] * boundary_suffix;
      if (rank < fitted_ranks) {
        const uint32_t new_index = new_base + rank * kRows + row;
        const float new_basis =
            transpose == 0U
                ? static_cast<float>(new_left_.GetValue(new_index))
                : static_cast<float>(new_right_.GetValue(new_index));
        value -= new_basis * coefficients[kRanks + rank];
      }
    }
    for (uint32_t record = 0; record < kWindow; ++record) {
      const uint32_t index = (state * kWindow + record) * kRows + row;
      const float output_basis =
          transpose == 0U
              ? static_cast<float>(record_left_.GetValue(index))
              : static_cast<float>(record_right_.GetValue(index));
      value += output_basis * record_coefficients[record];
    }
    return value;
  }

 private:
  uint32_t states_ = 0;
  AscendC::GlobalTensor<int32_t> projected_;
  AscendC::GlobalTensor<float> dynamic_scale_, smoothing_, scales_,
      record_decay_;
  AscendC::GlobalTensor<half> input_, old_left_, old_right_, record_left_,
      record_right_, new_left_, new_right_;
};
}  // namespace

extern "C" __global__ __aicore__ void
statecentric_leapquant_prepare_int8_vector(
    GM_ADDR input, GM_ADDR axis_scale, GM_ADDR quantized,
    GM_ADDR dynamic_scale, uint32_t states) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scalar_buffer;
  pipe.InitBuffer(scalar_buffer, 32);
  AscendC::GlobalTensor<half> input_global;
  AscendC::GlobalTensor<float> axis_scale_global;
  AscendC::GlobalTensor<int8_t> quantized_global;
  AscendC::GlobalTensor<float> dynamic_scale_global;
  input_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input),
                               states * kRows * kColumns);
  axis_scale_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(axis_scale),
                                    states * kRows);
  quantized_global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized),
                                   states * kRows * kColumns);
  dynamic_scale_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(dynamic_scale), states * kScalarStride);
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    const uint32_t vector_base = state * kRows * kColumns;
    const uint32_t scale_base = state * kRows;
    float maximum = 0.0F;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float value =
          static_cast<float>(input_global.GetValue(vector_base +
                                                   row * kColumns)) *
          axis_scale_global.GetValue(scale_base + row);
      const float magnitude = Absolute(value);
      maximum = magnitude > maximum ? magnitude : maximum;
    }
    const float scale = maximum > 1.0e-20F ? maximum / 127.0F : 1.0F;
    auto scale_output = scalar_buffer.Get<float>();
    scale_output.SetValue(0, scale);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams scale_copy{1, sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(dynamic_scale_global[state * kScalarStride],
                        scale_output, scale_copy);
    for (uint32_t row = 0; row < kRows; ++row) {
      const float value =
          static_cast<float>(input_global.GetValue(vector_base +
                                                   row * kColumns)) *
          axis_scale_global.GetValue(scale_base + row) / scale;
      float rounded = value + (value >= 0.0F ? 0.5F : -0.5F);
      rounded = rounded > 127.0F ? 127.0F : rounded;
      rounded = rounded < -127.0F ? -127.0F : rounded;
      quantized_global.SetValue(vector_base + row * kColumns,
                                static_cast<int8_t>(rounded));
      for (uint32_t column = 1; column < kColumns; ++column) {
        quantized_global.SetValue(vector_base + row * kColumns + column, 0);
      }
    }
  }
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_prepare_decode_pair(
    GM_ADDR reads, GM_ADDR queries, GM_ADDR decay, GM_ADDR smoothing,
    GM_ADDR quantized, GM_ADDR dynamic_scale, uint32_t states) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scale_buffer;
  pipe.InitBuffer(scale_buffer, 32);
  AscendC::GlobalTensor<half> reads_global, queries_global;
  AscendC::GlobalTensor<float> decay_global, smoothing_global,
      dynamic_scale_global;
  AscendC::GlobalTensor<int8_t> quantized_global;
  reads_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(reads),
                               states * kRows);
  queries_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queries),
                                 states * kRows);
  decay_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay), states);
  smoothing_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                                   states * kRows);
  quantized_global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized),
                                   states * kRows * kColumns);
  dynamic_scale_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(dynamic_scale), states * kScalarStride);
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    float maxima[2] = {0.0F, 0.0F};
    const uint32_t vector_base = state * kRows;
    const float current_decay = decay_global.GetValue(state);
    for (uint32_t row = 0; row < kRows; ++row) {
      const float smoothing_value =
          smoothing_global.GetValue(vector_base + row);
      const float pair[2] = {
          static_cast<float>(reads_global.GetValue(vector_base + row)) *
              smoothing_value,
          static_cast<float>(queries_global.GetValue(vector_base + row)) *
              current_decay * smoothing_value};
      for (uint32_t column = 0; column < 2; ++column) {
        const float magnitude = Absolute(pair[column]);
        maxima[column] = magnitude > maxima[column] ? magnitude : maxima[column];
      }
    }
    float pair_scale[4];
    for (uint32_t column = 0; column < 2; ++column) {
      pair_scale[column] =
          maxima[column] > 1.0e-20F ? maxima[column] / 127.0F : 1.0F;
    }
    float residual_maxima[2] = {0.0F, 0.0F};
    for (uint32_t row = 0; row < kRows; ++row) {
      const float smoothing_value =
          smoothing_global.GetValue(vector_base + row);
      const float pair[2] = {
          static_cast<float>(reads_global.GetValue(vector_base + row)) *
              smoothing_value,
          static_cast<float>(queries_global.GetValue(vector_base + row)) *
              current_decay * smoothing_value};
      for (uint32_t column = 0; column < 2; ++column) {
        float high = pair[column] / pair_scale[column];
        high += high >= 0.0F ? 0.5F : -0.5F;
        high = high > 127.0F ? 127.0F : high;
        high = high < -127.0F ? -127.0F : high;
        const float residual =
            pair[column] - static_cast<int8_t>(high) * pair_scale[column];
        const float magnitude = Absolute(residual);
        residual_maxima[column] = magnitude > residual_maxima[column]
                                       ? magnitude
                                       : residual_maxima[column];
      }
    }
    for (uint32_t column = 0; column < 2; ++column) {
      pair_scale[column + 2] = residual_maxima[column] > 1.0e-20F
                                   ? residual_maxima[column] / 127.0F
                                   : 1.0F;
    }
    auto scale_output = scale_buffer.Get<float>();
    for (uint32_t column = 0; column < 4; ++column) {
      scale_output.SetValue(column, pair_scale[column]);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams scale_copy{
        1, 4 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(
        dynamic_scale_global[state * kScalarStride], scale_output, scale_copy);
    const uint32_t output_base = state * kRows * kColumns;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float smoothing_value =
          smoothing_global.GetValue(vector_base + row);
      const float pair[2] = {
          static_cast<float>(reads_global.GetValue(vector_base + row)) *
              smoothing_value,
          static_cast<float>(queries_global.GetValue(vector_base + row)) *
              current_decay * smoothing_value};
      for (uint32_t column = 0; column < 2; ++column) {
        float high = pair[column] / pair_scale[column];
        high += high >= 0.0F ? 0.5F : -0.5F;
        high = high > 127.0F ? 127.0F : high;
        high = high < -127.0F ? -127.0F : high;
        const int8_t high_quantized = static_cast<int8_t>(high);
        float low = (pair[column] - high_quantized * pair_scale[column]) /
                    pair_scale[column + 2];
        low += low >= 0.0F ? 0.5F : -0.5F;
        low = low > 127.0F ? 127.0F : low;
        low = low < -127.0F ? -127.0F : low;
        const uint32_t high_column = column * 2;
        quantized_global.SetValue(output_base + row * kColumns + high_column,
                                  high_quantized);
        quantized_global.SetValue(
            output_base + row * kColumns + high_column + 1,
            static_cast<int8_t>(low));
      }
      for (uint32_t column = 4; column < kColumns; ++column) {
        quantized_global.SetValue(output_base + row * kColumns + column, 0);
      }
    }
  }
}

// Qwen3.5-35B has two value heads for every query/key head on each TP rank.
// Consume the model's native BF16 tensors directly so the Python integration
// does not launch repeat, multiply, cast, and contiguous kernels per layer.
extern "C" __global__ __aicore__ void
statecentric_leapquant_prepare_decode_pair_bf16(
    GM_ADDR keys, GM_ADDR queries, GM_ADDR decay, GM_ADDR smoothing,
    GM_ADDR quantized, GM_ADDR dynamic_scale, uint32_t states) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scale_buffer;
  pipe.InitBuffer(scale_buffer, 32);
  AscendC::GlobalTensor<uint16_t> keys_global, queries_global;
  AscendC::GlobalTensor<float> decay_global, smoothing_global,
      dynamic_scale_global;
  AscendC::GlobalTensor<int8_t> quantized_global;
  keys_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(keys),
                              states / 2 * kRows);
  queries_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(queries),
                                 states / 2 * kRows);
  decay_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay), states);
  smoothing_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                                   states * kRows);
  quantized_global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized),
                                   states * kRows * kColumns);
  dynamic_scale_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(dynamic_scale), states * kScalarStride);
  constexpr float query_scale = 0.08838834764831845F;
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    float maxima[2] = {0.0F, 0.0F};
    const uint32_t state_base = state * kRows;
    const uint32_t query_base = (state / 2) * kRows;
    auto scale_output = scale_buffer.Get<float>();
    scale_output.SetValue(0, decay_global.GetValue(state));
    AscendC::Exp(scale_output, scale_output, 1);
    AscendC::PipeBarrier<PIPE_V>();
    const float current_decay = scale_output.GetValue(0);
    for (uint32_t row = 0; row < kRows; ++row) {
      const float smoothing_value =
          smoothing_global.GetValue(state_base + row);
      const float pair[2] = {
          current_decay *
              BFloat16ToFloat(keys_global.GetValue(query_base + row)) *
              smoothing_value,
          query_scale * current_decay *
              BFloat16ToFloat(queries_global.GetValue(query_base + row)) *
              smoothing_value};
      for (uint32_t column = 0; column < 2; ++column) {
        const float magnitude = Absolute(pair[column]);
        maxima[column] = magnitude > maxima[column] ? magnitude : maxima[column];
      }
    }
    float pair_scale[4];
    for (uint32_t column = 0; column < 2; ++column) {
      pair_scale[column] =
          maxima[column] > 1.0e-20F ? maxima[column] / 127.0F : 1.0F;
    }
    float residual_maxima[2] = {0.0F, 0.0F};
    for (uint32_t row = 0; row < kRows; ++row) {
      const float smoothing_value =
          smoothing_global.GetValue(state_base + row);
      const float pair[2] = {
          current_decay *
              BFloat16ToFloat(keys_global.GetValue(query_base + row)) *
              smoothing_value,
          query_scale * current_decay *
              BFloat16ToFloat(queries_global.GetValue(query_base + row)) *
              smoothing_value};
      for (uint32_t column = 0; column < 2; ++column) {
        float high = pair[column] / pair_scale[column];
        high += high >= 0.0F ? 0.5F : -0.5F;
        high = high > 127.0F ? 127.0F : high;
        high = high < -127.0F ? -127.0F : high;
        const float residual =
            pair[column] - static_cast<int8_t>(high) * pair_scale[column];
        const float magnitude = Absolute(residual);
        residual_maxima[column] = magnitude > residual_maxima[column]
                                       ? magnitude
                                       : residual_maxima[column];
      }
    }
    for (uint32_t column = 0; column < 2; ++column) {
      pair_scale[column + 2] = residual_maxima[column] > 1.0e-20F
                                   ? residual_maxima[column] / 127.0F
                                   : 1.0F;
    }
    for (uint32_t column = 0; column < 4; ++column) {
      scale_output.SetValue(column, pair_scale[column]);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams scale_copy{
        1, 4 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(dynamic_scale_global[state * kScalarStride],
                        scale_output, scale_copy);
    const uint32_t output_base = state * kRows * kColumns;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float smoothing_value =
          smoothing_global.GetValue(state_base + row);
      const float pair[2] = {
          current_decay *
              BFloat16ToFloat(keys_global.GetValue(query_base + row)) *
              smoothing_value,
          query_scale * current_decay *
              BFloat16ToFloat(queries_global.GetValue(query_base + row)) *
              smoothing_value};
      for (uint32_t column = 0; column < 2; ++column) {
        float high = pair[column] / pair_scale[column];
        high += high >= 0.0F ? 0.5F : -0.5F;
        high = high > 127.0F ? 127.0F : high;
        high = high < -127.0F ? -127.0F : high;
        const int8_t high_quantized = static_cast<int8_t>(high);
        float low = (pair[column] - high_quantized * pair_scale[column]) /
                    pair_scale[column + 2];
        low += low >= 0.0F ? 0.5F : -0.5F;
        low = low > 127.0F ? 127.0F : low;
        low = low < -127.0F ? -127.0F : low;
        const uint32_t high_column = column * 2;
        quantized_global.SetValue(output_base + row * kColumns + high_column,
                                  high_quantized);
        quantized_global.SetValue(output_base + row * kColumns +
                                      high_column + 1,
                                  static_cast<int8_t>(low));
      }
      for (uint32_t column = 4; column < kColumns; ++column) {
        quantized_global.SetValue(output_base + row * kColumns + column, 0);
      }
    }
  }
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_finish_decode_pair(
    GM_ADDR projected, GM_ADDR dynamic_scale, GM_ADDR scales,
    GM_ADDR compensator_keys, GM_ADDR compensator_values,
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
    GM_ADDR positions, GM_ADDR decay, GM_ADDR write_keys, GM_ADDR reads,
    GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> position_buffer;
  pipe.InitBuffer(position_buffer, 32);
  AscendC::GlobalTensor<int32_t> projected_global;
  AscendC::GlobalTensor<float> dynamic_scale_global, scales_global,
      record_decay_global, decay_global, outputs_global;
  AscendC::GlobalTensor<half> compensator_keys_global,
      compensator_values_global, record_keys_global,
      record_corrections_global, write_keys_global, reads_global,
      queries_global, values_global;
  AscendC::GlobalTensor<uint32_t> positions_global;
  projected_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(projected),
                                   states * kRows * kColumns);
  dynamic_scale_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(dynamic_scale), states * kScalarStride);
  scales_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                                states * kRows);
  compensator_keys_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ half*>(compensator_keys),
      states * kRows * kRanks);
  compensator_values_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ half*>(compensator_values),
      states * kRows * kRanks);
  record_decay_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(record_decay), states * kWindow);
  record_keys_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_keys),
                                     states * kWindow * kRows);
  record_corrections_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ half*>(record_corrections),
      states * kWindow * kRows);
  positions_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(positions),
                                   states * kScalarStride);
  decay_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay), states);
  write_keys_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(write_keys),
                                    states * kRows);
  reads_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(reads),
                               states * kRows);
  queries_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queries),
                                 states * kRows);
  values_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(values),
                                states * kRows);
  outputs_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputs),
                                 states * kRows);
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    const uint32_t position = positions_global.GetValue(state * kScalarStride);
    if (position >= kWindow) continue;
    const uint32_t vector_base = state * kRows;
    const uint32_t compensator_base = state * kRows * kRanks;
    float old_projection[2][kRanks] = {};
    float record_projection[2][kWindow] = {};
    float suffix[kWindow];
    const float current_decay = decay_global.GetValue(state);
    float update_projection = 0.0F;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float read = static_cast<float>(reads_global.GetValue(vector_base + row));
      const float query = static_cast<float>(queries_global.GetValue(vector_base + row));
      update_projection +=
          static_cast<float>(write_keys_global.GetValue(vector_base + row)) * query;
      for (uint32_t rank = 0; rank < kRanks; ++rank) {
        const float key = static_cast<float>(compensator_keys_global.GetValue(
            compensator_base + row * kRanks + rank));
        old_projection[0][rank] += key * read;
        old_projection[1][rank] += key * query * current_decay;
      }
      for (uint32_t record = 0; record < position; ++record) {
        const float key = static_cast<float>(record_keys_global.GetValue(
            (state * kWindow + record) * kRows + row));
        record_projection[0][record] += key * read;
        record_projection[1][record] += key * query * current_decay;
      }
    }
    float boundary_suffix = 1.0F;
    for (uint32_t reverse = position; reverse > 0; --reverse) {
      const uint32_t record = reverse - 1;
      suffix[record] = boundary_suffix;
      boundary_suffix *=
          record_decay_global.GetValue(state * kWindow + record);
    }
    const uint32_t projected_base = state * kRows * kColumns;
    const uint32_t record_base = (state * kWindow + position) * kRows;
    for (uint32_t column = 0; column < kRows; ++column) {
      float state_read =
          (static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns)) *
               dynamic_scale_global.GetValue(state * kScalarStride) +
           static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns + 1)) *
               dynamic_scale_global.GetValue(state * kScalarStride + 2)) /
                         127.0F * scales_global.GetValue(vector_base + column) *
                         boundary_suffix;
      float output_read =
          (static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns + 2)) *
               dynamic_scale_global.GetValue(state * kScalarStride + 1) +
           static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns + 3)) *
               dynamic_scale_global.GetValue(state * kScalarStride + 3)) /
                          127.0F * scales_global.GetValue(vector_base + column) *
                          boundary_suffix;
      for (uint32_t rank = 0; rank < kRanks; ++rank) {
        const float basis = static_cast<float>(compensator_values_global.GetValue(
            compensator_base + column * kRanks + rank));
        state_read += basis * old_projection[0][rank] * boundary_suffix;
        output_read += basis * old_projection[1][rank] * boundary_suffix;
      }
      for (uint32_t record = 0; record < position; ++record) {
        const float basis = static_cast<float>(record_corrections_global.GetValue(
            (state * kWindow + record) * kRows + column));
        state_read += basis * record_projection[0][record] * suffix[record];
        output_read += basis * record_projection[1][record] * suffix[record];
      }
      const float correction =
          static_cast<float>(values_global.GetValue(vector_base + column)) -
          state_read;
      outputs_global.SetValue(vector_base + column,
                              output_read + correction * update_projection);
      record_keys_global.SetValue(
          record_base + column,
          write_keys_global.GetValue(vector_base + column));
      record_corrections_global.SetValue(record_base + column,
                                         static_cast<half>(correction));
    }
    record_decay_global.SetValue(state * kWindow + position, current_decay);
    auto position_output = position_buffer.Get<uint32_t>();
    position_output.SetValue(0, position + 1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams position_copy{
        1, sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(positions_global[state * kScalarStride],
                        position_output, position_copy);
  }
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_finish_decode_pair_bf16(
    GM_ADDR projected, GM_ADDR dynamic_scale, GM_ADDR scales,
    GM_ADDR compensator_keys, GM_ADDR compensator_values,
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
    GM_ADDR positions, GM_ADDR decay, GM_ADDR beta, GM_ADDR keys,
    GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> position_buffer;
  pipe.InitBuffer(position_buffer, 32);
  AscendC::GlobalTensor<int32_t> projected_global;
  AscendC::GlobalTensor<float> dynamic_scale_global, scales_global,
      record_decay_global, decay_global;
  AscendC::GlobalTensor<half> compensator_keys_global,
      compensator_values_global, record_keys_global,
      record_corrections_global;
  AscendC::GlobalTensor<uint16_t> beta_global, keys_global, queries_global,
      values_global, outputs_global;
  AscendC::GlobalTensor<uint32_t> positions_global;
  projected_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(projected),
                                   states * kRows * kColumns);
  dynamic_scale_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(dynamic_scale), states * kScalarStride);
  scales_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                                states * kRows);
  compensator_keys_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ half*>(compensator_keys),
      states * kRows * kRanks);
  compensator_values_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ half*>(compensator_values),
      states * kRows * kRanks);
  record_decay_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(record_decay), states * kWindow);
  record_keys_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_keys),
                                     states * kWindow * kRows);
  record_corrections_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ half*>(record_corrections),
      states * kWindow * kRows);
  positions_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(positions),
                                   states * kScalarStride);
  decay_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay), states);
  beta_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(beta),
                              states);
  keys_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(keys),
                              states / 2 * kRows);
  queries_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(queries),
                                 states / 2 * kRows);
  values_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(values),
                                states * kRows);
  outputs_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(outputs),
                                 states * kRows);
  constexpr float query_scale = 0.08838834764831845F;
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    const uint32_t position = positions_global.GetValue(state * kScalarStride);
    if (position >= kWindow) continue;
    const uint32_t vector_base = state * kRows;
    const uint32_t query_base = (state / 2) * kRows;
    const uint32_t compensator_base = state * kRows * kRanks;
    auto decay_local = position_buffer.Get<float>();
    decay_local.SetValue(0, decay_global.GetValue(state));
    AscendC::Exp(decay_local, decay_local, 1);
    AscendC::PipeBarrier<PIPE_V>();
    const float current_decay = decay_local.GetValue(0);
    const float current_beta = BFloat16ToFloat(beta_global.GetValue(state));
    float old_projection[2][kRanks] = {};
    float record_projection[2][kWindow] = {};
    float suffix[kWindow];
    float update_projection = 0.0F;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float key =
          BFloat16ToFloat(keys_global.GetValue(query_base + row));
      const float query = query_scale *
          BFloat16ToFloat(queries_global.GetValue(query_base + row));
      const float read = current_decay * key;
      const float write_key = current_beta * key;
      update_projection += write_key * query;
      for (uint32_t rank = 0; rank < kRanks; ++rank) {
        const float basis = static_cast<float>(compensator_keys_global.GetValue(
            compensator_base + row * kRanks + rank));
        old_projection[0][rank] += basis * read;
        old_projection[1][rank] += basis * query * current_decay;
      }
      for (uint32_t record = 0; record < position; ++record) {
        const float basis = static_cast<float>(record_keys_global.GetValue(
            (state * kWindow + record) * kRows + row));
        record_projection[0][record] += basis * read;
        record_projection[1][record] += basis * query * current_decay;
      }
    }
    float boundary_suffix = 1.0F;
    for (uint32_t reverse = position; reverse > 0; --reverse) {
      const uint32_t record = reverse - 1;
      suffix[record] = boundary_suffix;
      boundary_suffix *=
          record_decay_global.GetValue(state * kWindow + record);
    }
    const uint32_t projected_base = state * kRows * kColumns;
    const uint32_t record_base = (state * kWindow + position) * kRows;
    for (uint32_t column = 0; column < kRows; ++column) {
      float state_read =
          (static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns)) *
               dynamic_scale_global.GetValue(state * kScalarStride) +
           static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns + 1)) *
               dynamic_scale_global.GetValue(state * kScalarStride + 2)) /
          127.0F * scales_global.GetValue(vector_base + column) *
          boundary_suffix;
      float output_read =
          (static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns + 2)) *
               dynamic_scale_global.GetValue(state * kScalarStride + 1) +
           static_cast<float>(projected_global.GetValue(
               projected_base + column * kColumns + 3)) *
               dynamic_scale_global.GetValue(state * kScalarStride + 3)) /
          127.0F * scales_global.GetValue(vector_base + column) *
          boundary_suffix;
      for (uint32_t rank = 0; rank < kRanks; ++rank) {
        const float basis = static_cast<float>(compensator_values_global.GetValue(
            compensator_base + column * kRanks + rank));
        state_read += basis * old_projection[0][rank] * boundary_suffix;
        output_read += basis * old_projection[1][rank] * boundary_suffix;
      }
      for (uint32_t record = 0; record < position; ++record) {
        const float basis = static_cast<float>(record_corrections_global.GetValue(
            (state * kWindow + record) * kRows + column));
        state_read += basis * record_projection[0][record] * suffix[record];
        output_read += basis * record_projection[1][record] * suffix[record];
      }
      const float correction =
          BFloat16ToFloat(values_global.GetValue(vector_base + column)) -
          state_read;
      outputs_global.SetValue(vector_base + column,
                              FloatToBFloat16(
                                  output_read + correction * update_projection));
      const float key =
          BFloat16ToFloat(keys_global.GetValue(query_base + column));
      record_keys_global.SetValue(record_base + column,
                                  static_cast<half>(current_beta * key));
      record_corrections_global.SetValue(record_base + column,
                                         static_cast<half>(correction));
    }
    record_decay_global.SetValue(state * kWindow + position, current_decay);
    auto position_output = position_buffer.Get<uint32_t>();
    position_output.SetValue(0, position + 1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams position_copy{
        1, sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(positions_global[state * kScalarStride],
                        position_output, position_copy);
  }
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_finish_encoded_projection(
    GM_ADDR projected, GM_ADDR dynamic_scale, GM_ADDR input,
    GM_ADDR smoothing, GM_ADDR scales, GM_ADDR old_left, GM_ADDR old_right,
    GM_ADDR record_decay, GM_ADDR record_left, GM_ADDR record_right,
    GM_ADDR new_left, GM_ADDR new_right, GM_ADDR output, uint32_t states,
    uint32_t fitted_ranks, uint32_t transpose) {
  EncodedProjection projection;
  projection.Init(projected, dynamic_scale, input, smoothing, scales, old_left,
                  old_right, record_decay, record_left, record_right, new_left,
                  new_right, states);
  AscendC::GlobalTensor<half> output_global;
  output_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output),
                                states * kRows * kColumns);
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    float coefficients[kRanks * 2] = {0.0F, 0.0F, 0.0F, 0.0F,
                                      0.0F, 0.0F, 0.0F, 0.0F};
    float record_coefficients[kWindow];
    const float boundary_suffix = projection.PrepareState(
        state, fitted_ranks, transpose, coefficients, record_coefficients);
    float values[kRows];
    float squared_norm = 0.0F;
    for (uint32_t row = 0; row < kRows; ++row) {
      values[row] = projection.Value(state, row, fitted_ranks, transpose,
                                     coefficients, record_coefficients,
                                     boundary_suffix);
      squared_norm += values[row] * values[row];
    }
    const float inverse = squared_norm > 1.0e-20F
                              ? 1.0F / SquareRoot(squared_norm)
                              : 0.0F;
    const uint32_t base = state * kRows * kColumns;
    for (uint32_t row = 0; row < kRows; ++row) {
      output_global.SetValue(base + row * kColumns,
                             static_cast<half>(values[row] * inverse));
      for (uint32_t column = 1; column < kColumns; ++column) {
        output_global.SetValue(base + row * kColumns + column,
                               static_cast<half>(0.0F));
      }
    }
  }
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_commit_encoded_rank(
    GM_ADDR projected, GM_ADDR dynamic_scale, GM_ADDR input,
    GM_ADDR smoothing, GM_ADDR scales, GM_ADDR old_left, GM_ADDR old_right,
    GM_ADDR record_decay, GM_ADDR record_left, GM_ADDR record_right,
    GM_ADDR new_left, GM_ADDR new_right, GM_ADDR corrected, uint32_t states,
    uint32_t rank) {
  EncodedProjection projection;
  projection.Init(projected, dynamic_scale, input, smoothing, scales, old_left,
                  old_right, record_decay, record_left, record_right, new_left,
                  new_right, states);
  AscendC::GlobalTensor<half> input_global;
  AscendC::GlobalTensor<half> left_global;
  AscendC::GlobalTensor<half> right_global;
  AscendC::GlobalTensor<float> corrected_global;
  input_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input),
                               states * kRows * kColumns);
  left_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(new_left),
                              states * kRanks * kRows);
  right_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(new_right),
                               states * kRanks * kRows);
  corrected_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(corrected),
                                   states * kRows * kColumns);
  for (uint32_t state = AscendC::GetBlockIdx(); state < states;
       state += AscendC::GetBlockNum()) {
    float coefficients[kRanks * 2] = {0.0F, 0.0F, 0.0F, 0.0F,
                                      0.0F, 0.0F, 0.0F, 0.0F};
    float record_coefficients[kWindow];
    const float boundary_suffix = projection.PrepareState(
        state, rank, 0U, coefficients, record_coefficients);
    const uint32_t vector_base = state * kRows * kColumns;
    const uint32_t basis_base = state * kRanks * kRows + rank * kRows;
    for (uint32_t row = 0; row < kRows; ++row) {
      const float value = projection.Value(state, row, rank, 0U, coefficients,
                                           record_coefficients,
                                           boundary_suffix);
      corrected_global.SetValue(vector_base + row * kColumns, value);
      left_global.SetValue(basis_base + row, static_cast<half>(value));
      right_global.SetValue(
          basis_base + row,
          input_global.GetValue(vector_base + row * kColumns));
    }
  }
}

extern "C" int32_t statecentric_leapquant_prepare_int8_vector_launch_v1(
    void* stream, const uint16_t* input, const float* axis_scale,
    int8_t* quantized, float* dynamic_scale, uint32_t states) {
  if (!stream || !input || !axis_scale || !quantized || !dynamic_scale ||
      states == 0 || states > 16384) return 1;
  statecentric_leapquant_prepare_int8_vector<<<64, nullptr, stream>>>(
      reinterpret_cast<half*>(const_cast<uint16_t*>(input)),
      const_cast<float*>(axis_scale), quantized, dynamic_scale, states);
  return 0;
}

extern "C" int32_t statecentric_leapquant_prepare_decode_pair_launch_v1(
    void* stream, const uint16_t* reads, const uint16_t* queries,
    const float* decay, const float* smoothing, int8_t* quantized,
    float* dynamic_scale, uint32_t states) {
  if (!stream || !reads || !queries || !decay || !smoothing || !quantized ||
      !dynamic_scale || states == 0 || states > 16384) return 1;
  statecentric_leapquant_prepare_decode_pair<<<64, nullptr, stream>>>(
      reinterpret_cast<half*>(const_cast<uint16_t*>(reads)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(queries)),
      const_cast<float*>(decay), const_cast<float*>(smoothing), quantized,
      dynamic_scale, states);
  return 0;
}

extern "C" int32_t statecentric_leapquant_finish_decode_pair_launch_v1(
    void* stream, const int32_t* projected, const float* dynamic_scale,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, const float* decay, const uint16_t* write_keys,
    const uint16_t* reads, const uint16_t* queries, const uint16_t* values,
    float* outputs, uint32_t states) {
  if (!stream || !projected || !dynamic_scale || !scales ||
      !compensator_keys || !compensator_values || !record_decay ||
      !record_keys || !record_corrections || !positions || !decay ||
      !write_keys || !reads || !queries || !values || !outputs ||
      states == 0 || states > 16384) return 1;
  statecentric_leapquant_finish_decode_pair<<<64, nullptr, stream>>>(
      const_cast<int32_t*>(projected), const_cast<float*>(dynamic_scale),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)),
      record_decay, reinterpret_cast<half*>(record_keys),
      reinterpret_cast<half*>(record_corrections), positions,
      const_cast<float*>(decay),
      reinterpret_cast<half*>(const_cast<uint16_t*>(write_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(reads)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(queries)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(values)), outputs, states);
  return 0;
}

extern "C" int32_t
statecentric_leapquant_prepare_decode_pair_bf16_launch_v1(
    void* stream, const uint16_t* keys, const uint16_t* queries,
    const float* decay, const float* smoothing, int8_t* quantized,
    float* dynamic_scale, uint32_t states) {
  if (!stream || !keys || !queries || !decay || !smoothing || !quantized ||
      !dynamic_scale || states == 0 || states > 16384 || states % 2 != 0)
    return 1;
  statecentric_leapquant_prepare_decode_pair_bf16<<<64, nullptr, stream>>>(
      const_cast<uint16_t*>(keys), const_cast<uint16_t*>(queries),
      const_cast<float*>(decay), const_cast<float*>(smoothing), quantized,
      dynamic_scale, states);
  return 0;
}

extern "C" int32_t
statecentric_leapquant_finish_decode_pair_bf16_launch_v1(
    void* stream, const int32_t* projected, const float* dynamic_scale,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, const float* decay, const uint16_t* beta,
    const uint16_t* keys, const uint16_t* queries, const uint16_t* values,
    uint16_t* outputs, uint32_t states) {
  if (!stream || !projected || !dynamic_scale || !scales ||
      !compensator_keys || !compensator_values || !record_decay ||
      !record_keys || !record_corrections || !positions || !decay || !beta ||
      !keys || !queries || !values || !outputs || states == 0 ||
      states > 16384 || states % 2 != 0)
    return 1;
  statecentric_leapquant_finish_decode_pair_bf16<<<64, nullptr, stream>>>(
      const_cast<int32_t*>(projected), const_cast<float*>(dynamic_scale),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)),
      record_decay, reinterpret_cast<half*>(record_keys),
      reinterpret_cast<half*>(record_corrections), positions,
      const_cast<float*>(decay),
      const_cast<uint16_t*>(beta), const_cast<uint16_t*>(keys),
      const_cast<uint16_t*>(queries), const_cast<uint16_t*>(values), outputs,
      states);
  return 0;
}

extern "C" int32_t
statecentric_leapquant_finish_encoded_projection_launch_v1(
    void* stream, const int32_t* projected, const float* dynamic_scale,
    const uint16_t* input, const float* smoothing, const float* scales,
    const uint16_t* old_left, const uint16_t* old_right,
    const float* record_decay, const uint16_t* record_left,
    const uint16_t* record_right, const uint16_t* new_left,
    const uint16_t* new_right, uint16_t* output, uint32_t states,
    uint32_t fitted_ranks, uint32_t transpose) {
  if (!stream || !projected || !dynamic_scale || !input || !smoothing ||
      !scales || !old_left || !old_right || !record_decay || !record_left ||
      !record_right || !new_left || !new_right || !output || states == 0 ||
      states > 16384 || fitted_ranks > kRanks || transpose > 1U) return 1;
  statecentric_leapquant_finish_encoded_projection<<<64, nullptr, stream>>>(
      const_cast<int32_t*>(projected), const_cast<float*>(dynamic_scale),
      reinterpret_cast<half*>(const_cast<uint16_t*>(input)),
      const_cast<float*>(smoothing), const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(old_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(old_right)),
      const_cast<float*>(record_decay),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_right)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(new_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(new_right)),
      reinterpret_cast<half*>(output), states, fitted_ranks, transpose);
  return 0;
}

extern "C" int32_t statecentric_leapquant_commit_encoded_rank_launch_v1(
    void* stream, const int32_t* projected, const float* dynamic_scale,
    const uint16_t* input, const float* smoothing, const float* scales,
    const uint16_t* old_left, const uint16_t* old_right,
    const float* record_decay, const uint16_t* record_left,
    const uint16_t* record_right, uint16_t* new_left, uint16_t* new_right,
    float* corrected, uint32_t states, uint32_t rank) {
  if (!stream || !projected || !dynamic_scale || !input || !smoothing ||
      !scales || !old_left || !old_right || !record_decay || !record_left ||
      !record_right || !new_left || !new_right || !corrected || states == 0 ||
      states > 16384 || rank >= kRanks) return 1;
  statecentric_leapquant_commit_encoded_rank<<<64, nullptr, stream>>>(
      const_cast<int32_t*>(projected), const_cast<float*>(dynamic_scale),
      reinterpret_cast<half*>(const_cast<uint16_t*>(input)),
      const_cast<float*>(smoothing), const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(old_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(old_right)),
      const_cast<float*>(record_decay),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_left)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_right)),
      reinterpret_cast<half*>(new_left), reinterpret_cast<half*>(new_right),
      corrected, states, rank);
  return 0;
}

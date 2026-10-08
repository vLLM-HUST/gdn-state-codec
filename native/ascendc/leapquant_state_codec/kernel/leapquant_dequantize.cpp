#include "kernel_operator.h"

#include "statecentric/leapquant_state_codec_kernel.h"

namespace {
constexpr std::uint32_t kKeys = 128;
constexpr std::uint32_t kValues = 128;
constexpr std::uint32_t kRank = 4;
constexpr std::uint32_t kElements = kKeys * kValues;
constexpr std::uint32_t kCompensatorElements = kKeys * kRank;

class DequantizeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR quantized, GM_ADDR smoothing, GM_ADDR scales, GM_ADDR keys, GM_ADDR values, GM_ADDR dense) {
    quantized_.SetGlobalBuffer(reinterpret_cast<__gm__ std::int8_t*>(quantized), kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing), kKeys);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales), kValues);
    keys_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keys), kCompensatorElements);
    values_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values), kCompensatorElements);
    dense_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dense), kElements);
    pipe_.InitBuffer(quantized_buffer_, kValues * sizeof(std::int8_t));
    pipe_.InitBuffer(smoothing_buffer_, 32);
    pipe_.InitBuffer(scales_buffer_, kValues * sizeof(float));
    pipe_.InitBuffer(keys_buffer_, 32);
    pipe_.InitBuffer(values_buffer_, kCompensatorElements * sizeof(float));
    pipe_.InitBuffer(dense_buffer_, kValues * sizeof(float));
  }
  __aicore__ inline void Process() {
    auto quantized = quantized_buffer_.Get<std::int8_t>();
    auto smoothing = smoothing_buffer_.Get<float>();
    auto scales = scales_buffer_.Get<float>();
    auto keys = keys_buffer_.Get<float>();
    auto values = values_buffer_.Get<float>();
    auto dense = dense_buffer_.Get<float>();
    const AscendC::DataCopyPadExtParams<std::int8_t> no_int8_pad{false, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> no_float_pad{false, 0, 0, 0.0F};
    const std::uint32_t row = AscendC::GetBlockIdx();
    const AscendC::DataCopyExtParams quantized_copy{1, kValues * sizeof(std::int8_t), 0, 0, 0};
    const AscendC::DataCopyExtParams scalar_copy{1, sizeof(float), 0, 0, 0};
    const AscendC::DataCopyExtParams vector_copy{1, kValues * sizeof(float), 0, 0, 0};
    const AscendC::DataCopyExtParams key_copy{1, kRank * sizeof(float), 0, 0, 0};
    const AscendC::DataCopyExtParams compensator_copy{1, kCompensatorElements * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(quantized, quantized_[row * kValues], quantized_copy, no_int8_pad);
    AscendC::DataCopyPad(smoothing, smoothing_[row], scalar_copy, no_float_pad);
    AscendC::DataCopyPad(scales, scales_, vector_copy, no_float_pad);
    AscendC::DataCopyPad(keys, keys_[row * kRank], key_copy, no_float_pad);
    AscendC::DataCopyPad(values, values_, compensator_copy, no_float_pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    for (std::uint32_t column = 0; column < kValues; ++column) {
      float value = static_cast<float>(quantized.GetValue(column)) / 127.0F * scales.GetValue(column) * smoothing.GetValue(0);
      for (std::uint32_t rank = 0; rank < kRank; ++rank) {
        value += keys.GetValue(rank) * values.GetValue(column * kRank + rank);
      }
      dense.SetValue(column, value);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams dense_copy{1, kValues * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(dense_[row * kValues], dense, dense_copy);
  }
 private:
  AscendC::GlobalTensor<std::int8_t> quantized_;
  AscendC::GlobalTensor<float> smoothing_;
  AscendC::GlobalTensor<float> scales_;
  AscendC::GlobalTensor<float> keys_;
  AscendC::GlobalTensor<float> values_;
  AscendC::GlobalTensor<float> dense_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> quantized_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> smoothing_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scales_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> keys_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> values_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> dense_buffer_;
};
}  // namespace

extern "C" __global__ __aicore__ void statecentric_leapquant_dequantize(GM_ADDR quantized, GM_ADDR smoothing, GM_ADDR scales, GM_ADDR keys, GM_ADDR values, GM_ADDR dense) {
  DequantizeKernel kernel;
  kernel.Init(quantized, smoothing, scales, keys, values, dense);
  kernel.Process();
}

extern "C" std::int32_t statecentric_leapquant_dequantize_launch_v1(void* stream, const std::int8_t* quantized, const float* smoothing, const float* scales, const float* keys, const float* values, float* dense) {
  if (stream == nullptr || quantized == nullptr || smoothing == nullptr || scales == nullptr || keys == nullptr || values == nullptr || dense == nullptr) return 1;
  statecentric_leapquant_dequantize<<<128, nullptr, stream>>>(const_cast<std::int8_t*>(quantized), const_cast<float*>(smoothing), const_cast<float*>(scales), const_cast<float*>(keys), const_cast<float*>(values), dense);
  return 0;
}

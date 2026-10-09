#include "kernel_operator.h"

#include "statecentric/gdn_state_codec_kernel.h"

namespace {
constexpr std::uint32_t kKeys = 128;
constexpr std::uint32_t kValues = 128;
constexpr std::uint32_t kElements = kKeys * kValues;
__aicore__ inline float Absolute(float value) { return value < 0.0F ? -value : value; }

class QuantizeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR residual, GM_ADDR smoothing, GM_ADDR quantized, GM_ADDR scales) {
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(residual), kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing), kKeys);
    quantized_.SetGlobalBuffer(reinterpret_cast<__gm__ std::int8_t*>(quantized), kElements);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales), kValues);
    // A 4-byte strided GM block is padded to one 32-byte UB data block.
    pipe_.InitBuffer(residual_buffer_, kKeys * 32);
    pipe_.InitBuffer(smoothing_buffer_, kKeys * sizeof(float));
    // Each 1-byte UB-to-GM strided block likewise starts on a 32-byte boundary.
    pipe_.InitBuffer(quantized_buffer_, kKeys * 32);
    pipe_.InitBuffer(scales_buffer_, 32);
  }
  __aicore__ inline void Process() {
    auto residual = residual_buffer_.Get<float>();
    auto smoothing = smoothing_buffer_.Get<float>();
    auto quantized = quantized_buffer_.Get<std::int8_t>();
    auto scales = scales_buffer_.Get<float>();
    const std::uint32_t column = AscendC::GetBlockIdx();
    const AscendC::DataCopyExtParams residual_copy{kKeys, sizeof(float), (kValues - 1) * sizeof(float), 0, 0};
    const AscendC::DataCopyExtParams smoothing_copy{1, kKeys * sizeof(float), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> no_pad{false, 0, 0, 0.0F};
    AscendC::DataCopyPad(residual, residual_[column], residual_copy, no_pad);
    AscendC::DataCopyPad(smoothing, smoothing_, smoothing_copy, no_pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    float maximum = 0.0F;
    for (std::uint32_t row = 0; row < kKeys; ++row) {
      const float value = Absolute(residual.GetValue(row * 8) / smoothing.GetValue(row));
      maximum = value > maximum ? value : maximum;
    }
    scales.SetValue(0, maximum);
    for (std::uint32_t row = 0; row < kKeys; ++row) {
      const float scaled = maximum == 0.0F ? 0.0F : residual.GetValue(row * 8) / smoothing.GetValue(row) / maximum * 127.0F;
      float rounded = scaled >= 0.0F ? scaled + 0.5F : scaled - 0.5F;
      rounded = rounded > 127.0F ? 127.0F : rounded;
      rounded = rounded < -127.0F ? -127.0F : rounded;
      quantized.SetValue(row * 32, static_cast<std::int8_t>(rounded));
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams quantized_copy{kKeys, sizeof(std::int8_t), 0, kValues - 1, 0};
    const AscendC::DataCopyExtParams scales_copy{1, sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(quantized_[column], quantized, quantized_copy);
    AscendC::DataCopyPad(scales_[column], scales, scales_copy);
  }
 private:
  AscendC::GlobalTensor<float> residual_;
  AscendC::GlobalTensor<float> smoothing_;
  AscendC::GlobalTensor<std::int8_t> quantized_;
  AscendC::GlobalTensor<float> scales_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> residual_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> smoothing_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> quantized_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scales_buffer_;
};

class ProviderQuantizeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR provider_state, GM_ADDR smoothing,
                              GM_ADDR quantized, GM_ADDR scales,
                              std::uint32_t states) {
    states_ = states;
    provider_state_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(provider_state), states * kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                               states * kKeys);
    quantized_.SetGlobalBuffer(reinterpret_cast<__gm__ std::int8_t*>(quantized),
                               states * kElements);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                            states * kValues);
    pipe_.InitBuffer(row_buffer_, kKeys * sizeof(float));
    pipe_.InitBuffer(smoothing_buffer_, kKeys * sizeof(float));
    pipe_.InitBuffer(absolute_buffer_, kKeys * sizeof(float));
    pipe_.InitBuffer(half_buffer_, kKeys * sizeof(half));
    pipe_.InitBuffer(quantized_buffer_, kKeys * sizeof(std::int8_t));
    pipe_.InitBuffer(scale_buffer_, 32);
  }

  __aicore__ inline void Process() {
    auto row = row_buffer_.Get<float>();
    auto smoothing = smoothing_buffer_.Get<float>();
    auto absolute = absolute_buffer_.Get<float>();
    auto half_values = half_buffer_.Get<half>();
    auto quantized = quantized_buffer_.Get<std::int8_t>();
    auto reduction = scale_buffer_.Get<float>();
    const event_t vector_to_scalar = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::V_S));
    const event_t scalar_to_vector = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::S_V));
    const event_t quantized_reusable = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::MTE3_V));
    const std::uint32_t tasks = states_ * kValues;
    for (std::uint32_t task = AscendC::GetBlockIdx(); task < tasks;
         task += AscendC::GetBlockNum()) {
      const std::uint32_t state = task / kValues;
      const std::uint32_t value = task % kValues;
      const std::uint32_t state_base = state * kElements;
      const AscendC::DataCopyExtParams row_copy{
          1, kKeys * sizeof(float), 0, 0, 0};
      const AscendC::DataCopyExtParams smoothing_copy{
          1, kKeys * sizeof(float), 0, 0, 0};
      const AscendC::DataCopyPadExtParams<float> no_pad{false, 0, 0, 0.0F};
      AscendC::DataCopyPad(row,
                          provider_state_[state_base + value * kKeys],
                          row_copy, no_pad);
      AscendC::DataCopyPad(smoothing, smoothing_[state * kKeys],
                          smoothing_copy, no_pad);
      const event_t input_ready = static_cast<event_t>(
          pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_ready);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_ready);
      AscendC::Div(row, row, smoothing, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Abs(absolute, row, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceMax(
          reduction, absolute, 64, 2, 1, 1, 8,
          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceMax(
          reduction, reduction, 2, 1, 1, 1, 8,
          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
      const float maximum = reduction.GetValue(0);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
      AscendC::Muls(row, row, maximum > 0.0F ? 127.0F / maximum : 0.0F,
                    kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Maxs(row, row, -127.0F, kKeys);
      AscendC::Mins(row, row, 127.0F, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(half_values, row, AscendC::RoundMode::CAST_NONE, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(quantized, half_values,
                    AscendC::RoundMode::CAST_RINT, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      const event_t output_ready = static_cast<event_t>(
          pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(output_ready);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(output_ready);
      AscendC::DataCopy(quantized_[state_base + value * kKeys], quantized,
                        kKeys);
      const AscendC::DataCopyExtParams scale_copy{
          1, sizeof(float), 0, 0, 0};
      AscendC::DataCopyPad(scales_[state * kValues + value], reduction,
                          scale_copy);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(quantized_reusable);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(quantized_reusable);
    }
  }

 private:
  std::uint32_t states_ = 0;
  AscendC::GlobalTensor<float> provider_state_, smoothing_, scales_;
  AscendC::GlobalTensor<std::int8_t> quantized_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> row_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> smoothing_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> absolute_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> half_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> quantized_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scale_buffer_;
};
}  // namespace

extern "C" __global__ __aicore__ void statecentric_gdn_state_codec_quantize(GM_ADDR residual, GM_ADDR smoothing, GM_ADDR quantized, GM_ADDR scales) {
  QuantizeKernel kernel;
  kernel.Init(residual, smoothing, quantized, scales);
  kernel.Process();
}

extern "C" std::int32_t statecentric_gdn_state_codec_quantize_launch_v1(void* stream, const float* residual, const float* smoothing, std::int8_t* quantized, float* scales) {
  if (stream == nullptr || residual == nullptr || smoothing == nullptr || quantized == nullptr || scales == nullptr) return 1;
  statecentric_gdn_state_codec_quantize<<<128, nullptr, stream>>>(const_cast<float*>(residual), const_cast<float*>(smoothing), quantized, scales);
  return 0;
}

extern "C" __global__ __aicore__ void
statecentric_gdn_state_codec_provider_quantize(
    GM_ADDR provider_state, GM_ADDR smoothing, GM_ADDR quantized,
    GM_ADDR scales, uint32_t states) {
  ProviderQuantizeKernel kernel;
  kernel.Init(provider_state, smoothing, quantized, scales, states);
  kernel.Process();
}

extern "C" std::int32_t
statecentric_gdn_state_codec_provider_quantize_launch_v2(
    void* stream, const float* provider_state, const float* smoothing,
    std::int8_t* quantized, float* scales, std::uint32_t states) {
  if (stream == nullptr || provider_state == nullptr || smoothing == nullptr ||
      quantized == nullptr || scales == nullptr || states == 0 ||
      states > 64) {
    return 1;
  }
  const std::uint32_t blocks = states * 4U < 64U ? states * 4U : 64U;
  statecentric_gdn_state_codec_provider_quantize<<<blocks, nullptr, stream>>>(
      const_cast<float*>(provider_state),
      const_cast<float*>(smoothing), quantized, scales, states);
  return 0;
}

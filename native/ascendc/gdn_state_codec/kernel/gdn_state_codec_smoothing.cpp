#include "kernel_operator.h"

#include "statecentric/gdn_state_codec_kernel.h"

namespace {
constexpr std::uint32_t kKeys = 128;
constexpr std::uint32_t kValues = 128;
constexpr std::uint32_t kElements = kKeys * kValues;
constexpr float kFloor = 1.0e-8F;

__aicore__ inline float Absolute(float value) { return value < 0.0F ? -value : value; }
__aicore__ inline float SquareRoot(float value) {
  float estimate = value > 1.0F ? value : 1.0F;
  for (std::uint32_t iteration = 0; iteration < 20; ++iteration) {
    estimate = 0.5F * (estimate + value / estimate);
  }
  return estimate;
}

class SmoothingKernel {
 public:
  __aicore__ inline void Init(GM_ADDR residual, GM_ADDR smoothing) {
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(residual), kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing), kKeys);
    pipe_.InitBuffer(residual_buffer_, kValues * sizeof(float));
    pipe_.InitBuffer(smoothing_buffer_, 32);
  }
  __aicore__ inline void Process() {
    auto residual = residual_buffer_.Get<float>();
    auto smoothing = smoothing_buffer_.Get<float>();
    const std::uint32_t row = AscendC::GetBlockIdx();
    const AscendC::DataCopyExtParams residual_copy{1, kValues * sizeof(float), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> no_pad{false, 0, 0, 0.0F};
    AscendC::DataCopyPad(residual, residual_[row * kValues], residual_copy, no_pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < kValues; ++column) {
      sum += Absolute(residual.GetValue(column));
    }
    const float mean = sum / 128.0F;
    smoothing.SetValue(0, SquareRoot(mean > kFloor ? mean : kFloor));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams smoothing_copy{1, sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(smoothing_[row], smoothing, smoothing_copy);
  }
 private:
  AscendC::GlobalTensor<float> residual_;
  AscendC::GlobalTensor<float> smoothing_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> residual_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> smoothing_buffer_;
};

class ProviderSmoothingKernel {
 public:
  __aicore__ inline void Init(GM_ADDR provider_state, GM_ADDR smoothing,
                              std::uint32_t states) {
    states_ = states;
    provider_state_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(provider_state), states * kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                               states * kKeys);
    pipe_.InitBuffer(row_buffer_, kKeys * sizeof(float));
    pipe_.InitBuffer(accumulator_buffer_, kKeys * sizeof(float));
  }

  __aicore__ inline void Process() {
    auto row = row_buffer_.Get<float>();
    auto accumulator = accumulator_buffer_.Get<float>();
    const event_t row_reusable = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
    for (std::uint32_t state = AscendC::GetBlockIdx(); state < states_;
         state += AscendC::GetBlockNum()) {
      const std::uint32_t state_base = state * kElements;
      AscendC::Duplicate(accumulator, 0.0F, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      const AscendC::DataCopyExtParams row_copy{
          1, kKeys * sizeof(float), 0, 0, 0};
      const AscendC::DataCopyPadExtParams<float> no_pad{false, 0, 0, 0.0F};
      for (std::uint32_t value = 0; value < kValues; ++value) {
        AscendC::DataCopyPad(row,
                            provider_state_[state_base + value * kKeys],
                            row_copy, no_pad);
        const event_t input_ready = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_ready);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_ready);
        AscendC::Abs(row, row, kKeys);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(accumulator, accumulator, row, kKeys);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(row_reusable);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(row_reusable);
      }
      AscendC::Muls(accumulator, accumulator, 1.0F / 128.0F, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Maxs(accumulator, accumulator, kFloor, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sqrt(accumulator, accumulator, kKeys);
      AscendC::PipeBarrier<PIPE_V>();
      const event_t output_ready = static_cast<event_t>(
          pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(output_ready);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(output_ready);
      AscendC::DataCopy(smoothing_[state * kKeys], accumulator, kKeys);
    }
  }

 private:
  std::uint32_t states_ = 0;
  AscendC::GlobalTensor<float> provider_state_, smoothing_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> row_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> accumulator_buffer_;
};
}  // namespace

extern "C" __global__ __aicore__ void statecentric_gdn_state_codec_smoothing(GM_ADDR residual, GM_ADDR smoothing) {
  SmoothingKernel kernel;
  kernel.Init(residual, smoothing);
  kernel.Process();
}

extern "C" std::int32_t statecentric_gdn_state_codec_smoothing_launch_v1(void* stream, const float* residual, float* smoothing) {
  if (stream == nullptr || residual == nullptr || smoothing == nullptr) return 1;
  statecentric_gdn_state_codec_smoothing<<<128, nullptr, stream>>>(const_cast<float*>(residual), smoothing);
  return 0;
}

extern "C" __global__ __aicore__ void
statecentric_gdn_state_codec_provider_smoothing(
    GM_ADDR provider_state, GM_ADDR smoothing, uint32_t states) {
  ProviderSmoothingKernel kernel;
  kernel.Init(provider_state, smoothing, states);
  kernel.Process();
}

extern "C" std::int32_t
statecentric_gdn_state_codec_provider_smoothing_launch_v2(
    void* stream, const float* provider_state, float* smoothing,
    std::uint32_t states) {
  if (stream == nullptr || provider_state == nullptr || smoothing == nullptr ||
      states == 0 || states > 64) {
    return 1;
  }
  const std::uint32_t blocks = states;
  statecentric_gdn_state_codec_provider_smoothing<<<blocks, nullptr, stream>>>(
      const_cast<float*>(provider_state), smoothing, states);
  return 0;
}

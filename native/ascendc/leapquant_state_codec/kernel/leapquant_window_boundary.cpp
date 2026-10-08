#include "kernel_operator.h"

#include "statecentric/leapquant_windowed_gdn_kernel.h"

namespace {
constexpr uint32_t kD = 128;
constexpr uint32_t kElements = kD * kD;
constexpr uint32_t kRank = 4;
constexpr uint32_t kWindow = 16;
constexpr uint32_t kPositionStride = 8;
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

__aicore__ inline bool Normalize(float* vector) {
  float squared_norm = 0.0F;
  for (uint32_t axis = 0; axis < kD; ++axis) {
    squared_norm += vector[axis] * vector[axis];
  }
  if (squared_norm <= 1.0e-20F) return false;
  const float inverse_norm = 1.0F / SquareRoot(squared_norm);
  for (uint32_t axis = 0; axis < kD; ++axis) {
    vector[axis] *= inverse_norm;
  }
  return true;
}

class WindowBoundaryKernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
      GM_ADDR compensator_keys, GM_ADDR compensator_values,
      GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
      GM_ADDR positions, GM_ADDR next_residual, GM_ADDR next_smoothing,
      GM_ADDR next_scales, GM_ADDR next_compensator_keys,
      GM_ADDR next_compensator_values, GM_ADDR residual_workspace,
      uint32_t states,
      uint32_t power_iterations, uint32_t record_merge_ready) {
    states_ = states;
    power_iterations_ = power_iterations;
    record_merge_ready_ = record_merge_ready != 0U;
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(residual),
                              states * kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                               states * kD);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                            states * kD);
    compensator_keys_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(compensator_keys), states * kD * kRank);
    compensator_values_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(compensator_values),
        states * kD * kRank);
    record_decay_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(record_decay),
                                  states * kWindow);
    record_keys_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_keys),
                                 states * kWindow * kD);
    record_corrections_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(record_corrections),
        states * kWindow * kD);
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(positions),
                               states * kPositionStride);
    next_residual_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(next_residual), states * kElements);
    next_smoothing_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(next_smoothing), states * kD);
    next_scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(next_scales),
                                 states * kD);
    next_compensator_keys_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(next_compensator_keys),
        states * kD * kRank);
    next_compensator_values_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(next_compensator_values),
        states * kD * kRank);
    residual_workspace_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(residual_workspace),
        states * kElements);
    pipe_.InitBuffer(position_buffer_, 32);
    pipe_.InitBuffer(fast_residual_int8_buffer_, kD * sizeof(int8_t));
    pipe_.InitBuffer(fast_residual_half_buffer_, kD * sizeof(half));
    pipe_.InitBuffer(fast_residual_float_buffer_, kD * sizeof(float));
    pipe_.InitBuffer(fast_smoothing_buffer_, kD * sizeof(float));
    pipe_.InitBuffer(fast_record_half_buffer_,
                     kWindow * kD * sizeof(half));
    pipe_.InitBuffer(fast_record_keys_float_buffer_,
                     kWindow * kD * sizeof(float));
    pipe_.InitBuffer(fast_working_buffer_, kD * sizeof(float));
    pipe_.InitBuffer(fast_quantized_buffer_, kD * sizeof(int8_t));
  }

  __aicore__ inline void BuildSuffix(uint32_t state, float* record_suffix,
                                     float& boundary_suffix) {
    float suffix = 1.0F;
    for (uint32_t reverse = kWindow; reverse > 0; --reverse) {
      const uint32_t record = reverse - 1;
      record_suffix[record] = suffix;
      suffix *= record_decay_.GetValue(state * kWindow + record);
    }
    boundary_suffix = suffix;
  }

  __aicore__ inline void ApplyRight(
      uint32_t state, const float* vector, const float* record_suffix,
      float boundary_suffix, const float* new_keys, const float* new_values,
      uint32_t fitted_ranks, float* output) {
    float old_projection[kRank] = {0.0F, 0.0F, 0.0F, 0.0F};
    float record_projection[kWindow];
    float fitted_projection[kRank] = {0.0F, 0.0F, 0.0F, 0.0F};
    const uint32_t compensator_base = state * kD * kRank;
    for (uint32_t rank = 0; rank < kRank; ++rank) {
      for (uint32_t column = 0; column < kD; ++column) {
        old_projection[rank] +=
            static_cast<float>(compensator_values_.GetValue(
                compensator_base + column * kRank + rank)) * vector[column];
      }
    }
    for (uint32_t record = 0; record < kWindow; ++record) {
      record_projection[record] = 0.0F;
      const uint32_t base = (state * kWindow + record) * kD;
      for (uint32_t column = 0; column < kD; ++column) {
        record_projection[record] +=
            static_cast<float>(record_corrections_.GetValue(base + column)) *
            vector[column];
      }
    }
    for (uint32_t rank = 0; rank < fitted_ranks; ++rank) {
      for (uint32_t column = 0; column < kD; ++column) {
        fitted_projection[rank] +=
            new_values[column * kRank + rank] * vector[column];
      }
    }
    const uint32_t boundary_base = state * kElements;
    for (uint32_t row = 0; row < kD; ++row) {
      float value = 0.0F;
      for (uint32_t column = 0; column < kD; ++column) {
        value += static_cast<float>(residual_.GetValue(
                     boundary_base + column * kD + row)) /
                 127.0F * smoothing_.GetValue(state * kD + row) *
                 scales_.GetValue(state * kD + column) * vector[column] *
                 boundary_suffix;
      }
      for (uint32_t rank = 0; rank < kRank; ++rank) {
        value += static_cast<float>(compensator_keys_.GetValue(
                     compensator_base + row * kRank + rank)) *
                 old_projection[rank] * boundary_suffix;
      }
      for (uint32_t record = 0; record < kWindow; ++record) {
        value += static_cast<float>(record_keys_.GetValue(
                     (state * kWindow + record) * kD + row)) *
                 record_projection[record] * record_suffix[record];
      }
      for (uint32_t rank = 0; rank < fitted_ranks; ++rank) {
        value -= new_keys[row * kRank + rank] * fitted_projection[rank];
      }
      output[row] = value;
    }
  }

  __aicore__ inline void ApplyTranspose(
      uint32_t state, const float* vector, const float* record_suffix,
      float boundary_suffix, const float* new_keys, const float* new_values,
      uint32_t fitted_ranks, float* output) {
    float old_projection[kRank] = {0.0F, 0.0F, 0.0F, 0.0F};
    float record_projection[kWindow];
    float fitted_projection[kRank] = {0.0F, 0.0F, 0.0F, 0.0F};
    const uint32_t compensator_base = state * kD * kRank;
    for (uint32_t rank = 0; rank < kRank; ++rank) {
      for (uint32_t row = 0; row < kD; ++row) {
        old_projection[rank] +=
            static_cast<float>(compensator_keys_.GetValue(
                compensator_base + row * kRank + rank)) * vector[row];
      }
    }
    for (uint32_t record = 0; record < kWindow; ++record) {
      record_projection[record] = 0.0F;
      const uint32_t base = (state * kWindow + record) * kD;
      for (uint32_t row = 0; row < kD; ++row) {
        record_projection[record] +=
            static_cast<float>(record_keys_.GetValue(base + row)) * vector[row];
      }
    }
    for (uint32_t rank = 0; rank < fitted_ranks; ++rank) {
      for (uint32_t row = 0; row < kD; ++row) {
        fitted_projection[rank] +=
            new_keys[row * kRank + rank] * vector[row];
      }
    }
    const uint32_t boundary_base = state * kElements;
    for (uint32_t column = 0; column < kD; ++column) {
      float value = 0.0F;
      for (uint32_t row = 0; row < kD; ++row) {
        value += static_cast<float>(residual_.GetValue(
                     boundary_base + column * kD + row)) /
                 127.0F * smoothing_.GetValue(state * kD + row) *
                 scales_.GetValue(state * kD + column) * vector[row] *
                 boundary_suffix;
      }
      for (uint32_t rank = 0; rank < kRank; ++rank) {
        value += static_cast<float>(compensator_values_.GetValue(
                     compensator_base + column * kRank + rank)) *
                 old_projection[rank] * boundary_suffix;
      }
      for (uint32_t record = 0; record < kWindow; ++record) {
        value += static_cast<float>(record_corrections_.GetValue(
                     (state * kWindow + record) * kD + column)) *
                 record_projection[record] * record_suffix[record];
      }
      for (uint32_t rank = 0; rank < fitted_ranks; ++rank) {
        value -= new_values[column * kRank + rank] *
                 fitted_projection[rank];
      }
      output[column] = value;
    }
  }

  __aicore__ inline float ResidualCell(
      uint32_t state, uint32_t row, uint32_t column,
      const float* record_suffix, float boundary_suffix,
      const float* new_keys, const float* new_values) {
    const uint32_t boundary_base = state * kElements;
    const uint32_t compensator_base = state * kD * kRank;
    float value = static_cast<float>(residual_.GetValue(
                      boundary_base + column * kD + row)) /
                  127.0F * smoothing_.GetValue(state * kD + row) *
                  scales_.GetValue(state * kD + column) * boundary_suffix;
    for (uint32_t rank = 0; rank < kRank; ++rank) {
      value += static_cast<float>(compensator_keys_.GetValue(
                   compensator_base + row * kRank + rank)) *
               static_cast<float>(compensator_values_.GetValue(
                   compensator_base + column * kRank + rank)) *
               boundary_suffix;
      value -= new_keys[row * kRank + rank] *
               new_values[column * kRank + rank];
    }
    for (uint32_t record = 0; record < kWindow; ++record) {
      const uint32_t base = (state * kWindow + record) * kD;
      value += static_cast<float>(record_keys_.GetValue(base + row)) *
               static_cast<float>(record_corrections_.GetValue(base + column)) *
               record_suffix[record];
    }
    return value;
  }

  __aicore__ inline void ProcessState(uint32_t state) {
    if (positions_.GetValue(state * kPositionStride) != kWindow) return;
    float record_suffix[kWindow];
    float boundary_suffix = 1.0F;
    BuildSuffix(state, record_suffix, boundary_suffix);
    float new_keys[kD * kRank];
    float new_values[kD * kRank];
    for (uint32_t index = 0; index < kD * kRank; ++index) {
      new_keys[index] = 0.0F;
      new_values[index] = 0.0F;
    }
    float left[kD];
    float right[kD];
    const uint32_t fitted_rank_limit = power_iterations_ == 0 ? 0 : kRank;
    for (uint32_t rank = 0; rank < fitted_rank_limit; ++rank) {
      for (uint32_t column = 0; column < kD; ++column) {
        right[column] = static_cast<float>(
            static_cast<int32_t>(((rank + 1) * (column + 3)) % 17) - 8);
      }
      (void)Normalize(right);
      for (uint32_t iteration = 0; iteration < power_iterations_; ++iteration) {
        ApplyRight(state, right, record_suffix, boundary_suffix, new_keys,
                   new_values, rank, left);
        if (!Normalize(left)) break;
        ApplyTranspose(state, left, record_suffix, boundary_suffix, new_keys,
                       new_values, rank, right);
        if (!Normalize(right)) break;
      }
      ApplyRight(state, right, record_suffix, boundary_suffix, new_keys,
                 new_values, rank, left);
      float singular_value = 0.0F;
      for (uint32_t row = 0; row < kD; ++row) {
        singular_value += left[row] * left[row];
      }
      singular_value = SquareRoot(singular_value);
      if (singular_value > 1.0e-20F) {
        const float inverse = 1.0F / singular_value;
        for (uint32_t row = 0; row < kD; ++row) {
          const half stored = static_cast<half>(left[row] * inverse *
                                                singular_value);
          new_keys[row * kRank + rank] = static_cast<float>(stored);
        }
        for (uint32_t column = 0; column < kD; ++column) {
          const half stored = static_cast<half>(right[column]);
          new_values[column * kRank + rank] = static_cast<float>(stored);
        }
      }
    }

    for (uint32_t row = 0; row < kD; ++row) {
      float mean_absolute = 0.0F;
      for (uint32_t column = 0; column < kD; ++column) {
        const float value = ResidualCell(
            state, row, column, record_suffix, boundary_suffix, new_keys,
            new_values);
        residual_workspace_.SetValue(
            state * kElements + row * kD + column, value);
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
                state * kElements + row * kD + column) /
            next_smoothing_.GetValue(state * kD + row));
        maximum = value > maximum ? value : maximum;
      }
      next_scales_.SetValue(state * kD + column, maximum);
      for (uint32_t row = 0; row < kD; ++row) {
        const float smoothed = residual_workspace_.GetValue(
                                   state * kElements + row * kD + column) /
            next_smoothing_.GetValue(state * kD + row);
        float quantized = maximum == 0.0F ? 0.0F
                                         : smoothed / maximum * 127.0F;
        quantized += quantized >= 0.0F ? 0.5F : -0.5F;
        quantized = quantized > 127.0F ? 127.0F : quantized;
        quantized = quantized < -127.0F ? -127.0F : quantized;
        next_residual_.SetValue(state * kElements + column * kD + row,
                                static_cast<int8_t>(quantized));
      }
    }
    const uint32_t next_compensator_base = state * kD * kRank;
    for (uint32_t axis = 0; axis < kD; ++axis) {
      for (uint32_t rank = 0; rank < kRank; ++rank) {
        next_compensator_keys_.SetValue(
            next_compensator_base + axis * kRank + rank,
            static_cast<half>(new_keys[axis * kRank + rank]));
        next_compensator_values_.SetValue(
            next_compensator_base + axis * kRank + rank,
            static_cast<half>(new_values[axis * kRank + rank]));
      }
    }
    for (uint32_t record = 0; record < kWindow; ++record) {
      record_decay_.SetValue(state * kWindow + record, 0.0F);
      const uint32_t base = (state * kWindow + record) * kD;
      for (uint32_t axis = 0; axis < kD; ++axis) {
        record_keys_.SetValue(base + axis, static_cast<half>(0.0F));
        record_corrections_.SetValue(base + axis, static_cast<half>(0.0F));
      }
    }
    auto position_output = position_buffer_.Get<uint32_t>();
    position_output.SetValue(0, 0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const AscendC::DataCopyExtParams position_copy{
        1, sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(positions_[state * kPositionStride], position_output,
                        position_copy);
  }

  __aicore__ inline void ProcessQuantOnlyTask(uint32_t state,
                                               uint32_t shard) {
    constexpr uint32_t kShards = 2;
    constexpr uint32_t kColumnsPerShard = kD / kShards;
    const uint32_t column_begin = shard * kColumnsPerShard;
    const uint32_t column_end = column_begin + kColumnsPerShard;
    const uint32_t matrix_base = state * kElements;
    const uint32_t vector_base = state * kD;
    float record_suffix[kWindow];
    float boundary_suffix = 1.0F;
    BuildSuffix(state, record_suffix, boundary_suffix);

    auto residual_int8 = fast_residual_int8_buffer_.Get<int8_t>();
    auto residual_half = fast_residual_half_buffer_.Get<half>();
    auto residual_float = fast_residual_float_buffer_.Get<float>();
    auto smoothing = fast_smoothing_buffer_.Get<float>();
    auto record_half = fast_record_half_buffer_.Get<half>();
    auto record_keys_float = fast_record_keys_float_buffer_.Get<float>();
    auto working = fast_working_buffer_.Get<float>();
    auto quantized = fast_quantized_buffer_.Get<int8_t>();
    const event_t vector_to_scalar = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::V_S));
    const event_t scalar_to_vector = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::S_V));
    const event_t quantized_reusable = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::DataCopy(smoothing, smoothing_[vector_base], kD);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    if (!record_merge_ready_) {
      AscendC::DataCopy(
          record_half, record_keys_[state * kWindow * kD], kWindow * kD);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::Cast(record_keys_float, record_half,
                    AscendC::RoundMode::CAST_NONE, kWindow * kD);
      AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t column = column_begin; column < column_end; ++column) {
      AscendC::DataCopy(residual_int8,
                        residual_[matrix_base + column * kD], kD);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::Cast(residual_half, residual_int8,
                    AscendC::RoundMode::CAST_NONE, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(residual_float, residual_half,
                    AscendC::RoundMode::CAST_NONE, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(
          residual_float, residual_float,
          scales_.GetValue(vector_base + column) * boundary_suffix / 127.0F,
          kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(working, residual_float, smoothing, kD);
      AscendC::PipeBarrier<PIPE_V>();
      if (record_merge_ready_) {
        AscendC::DataCopy(
            record_keys_float,
            residual_workspace_[matrix_base + column * kD], kD);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Add(working, working, record_keys_float, kD);
        AscendC::PipeBarrier<PIPE_V>();
      } else {
        for (uint32_t record = 0; record < kWindow; ++record) {
          const uint32_t record_base =
              (state * kWindow + record) * kD;
          AscendC::Axpy(
              working, record_keys_float[record * kD],
              static_cast<float>(
                  record_corrections_.GetValue(record_base + column)) *
                  record_suffix[record],
              kD);
          AscendC::PipeBarrier<PIPE_V>();
        }
      }
      AscendC::Div(working, working, smoothing, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Abs(residual_float, working, kD);
      AscendC::PipeBarrier<PIPE_V>();
      auto reduction = position_buffer_.Get<float>();
      AscendC::WholeReduceMax(
          reduction, residual_float, 64, 2, 1, 1, 8,
          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceMax(
          reduction, reduction, 2, 1, 1, 1, 8,
          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
      const float maximum = reduction.GetValue(0);
      next_scales_.SetValue(vector_base + column, maximum);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
      AscendC::Muls(working, working,
                    maximum > 0.0F ? 127.0F / maximum : 0.0F, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Maxs(working, working, -127.0F, kD);
      AscendC::Mins(working, working, 127.0F, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(residual_half, working,
                    AscendC::RoundMode::CAST_NONE, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(quantized, residual_half,
                    AscendC::RoundMode::CAST_RINT, kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::DataCopy(next_residual_[matrix_base + column * kD],
                        quantized, kD);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(quantized_reusable);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(quantized_reusable);
    }
    AscendC::DataCopy(next_smoothing_[vector_base + column_begin],
                      smoothing[column_begin], kColumnsPerShard);
    constexpr uint32_t kCompensatorShardElements =
        kColumnsPerShard * kRank;
    AscendC::Duplicate(record_half, static_cast<half>(0.0F),
                       kCompensatorShardElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const uint32_t compensator_shard_base =
        state * kD * kRank + column_begin * kRank;
    AscendC::DataCopy(next_compensator_keys_[compensator_shard_base],
                      record_half, kCompensatorShardElements);
    AscendC::DataCopy(next_compensator_values_[compensator_shard_base],
                      record_half, kCompensatorShardElements);
  }

  __aicore__ inline void Process() {
    if (power_iterations_ == 0) {
      constexpr uint32_t kShards = 2;
      for (uint32_t task = AscendC::GetBlockIdx();
           task < states_ * kShards; task += AscendC::GetBlockNum()) {
        ProcessQuantOnlyTask(task / kShards, task % kShards);
      }
      return;
    }
    for (uint32_t state = AscendC::GetBlockIdx(); state < states_;
         state += AscendC::GetBlockNum()) {
      ProcessState(state);
    }
  }

 private:
  uint32_t states_ = 0;
  uint32_t power_iterations_ = 0;
  bool record_merge_ready_ = false;
  AscendC::GlobalTensor<int8_t> residual_, next_residual_;
  AscendC::GlobalTensor<float> smoothing_, scales_, record_decay_;
  AscendC::GlobalTensor<float> next_smoothing_, next_scales_;
  AscendC::GlobalTensor<half> compensator_keys_, compensator_values_;
  AscendC::GlobalTensor<half> record_keys_, record_corrections_;
  AscendC::GlobalTensor<half> next_compensator_keys_,
      next_compensator_values_;
  AscendC::GlobalTensor<float> residual_workspace_;
  AscendC::GlobalTensor<uint32_t> positions_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> position_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_residual_int8_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_residual_half_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_residual_float_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_smoothing_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_record_half_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC>
      fast_record_keys_float_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_working_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fast_quantized_buffer_;
};

class BoundaryRecordScaleKernel {
 public:
  __aicore__ inline void Init(GM_ADDR record_decay, GM_ADDR record_keys,
                              GM_ADDR scaled_record_keys,
                              uint32_t states) {
    states_ = states;
    record_decay_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(record_decay), states * kWindow);
    record_keys_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_keys),
                                 states * kWindow * kD);
    scaled_record_keys_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(scaled_record_keys),
        states * kWindow * kD);
    pipe_.InitBuffer(record_buffer_, kD * sizeof(half));
  }

  __aicore__ inline void Process() {
    const event_t copy_in_ready = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
    const event_t scalar_ready = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::S_V));
    const event_t copy_out_ready = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
    const event_t buffer_reusable = static_cast<event_t>(
        pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    for (uint32_t task = AscendC::GetBlockIdx();
         task < states_ * kWindow; task += AscendC::GetBlockNum()) {
      const uint32_t state = task / kWindow;
      const uint32_t record = task % kWindow;
      float suffix = 1.0F;
      for (uint32_t later = record + 1; later < kWindow; ++later) {
        suffix *= record_decay_.GetValue(state * kWindow + later);
      }
      auto local = record_buffer_.Get<half>();
      const uint32_t base = task * kD;
      AscendC::DataCopy(local, record_keys_[base], kD);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(copy_in_ready);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(copy_in_ready);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_ready);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_ready);
      AscendC::Muls(local, local, static_cast<half>(suffix), kD);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(copy_out_ready);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(copy_out_ready);
      AscendC::DataCopy(scaled_record_keys_[base], local, kD);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(buffer_reusable);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(buffer_reusable);
    }
  }

 private:
  uint32_t states_ = 0;
  AscendC::GlobalTensor<float> record_decay_;
  AscendC::GlobalTensor<half> record_keys_, scaled_record_keys_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> record_buffer_;
};

class WindowBoundaryFinalizeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR positions, uint32_t states) {
    states_ = states;
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(positions),
                               states * kPositionStride);
    pipe_.InitBuffer(position_buffer_, 32);
  }
  __aicore__ inline void Process() {
    for (uint32_t state = AscendC::GetBlockIdx(); state < states_;
         state += AscendC::GetBlockNum()) {
      auto position = position_buffer_.Get<uint32_t>();
      position.SetValue(0, 0);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      const AscendC::DataCopyExtParams copy{1, sizeof(uint32_t), 0, 0, 0};
      AscendC::DataCopyPad(positions_[state * kPositionStride], position,
                          copy);
    }
  }

 private:
  uint32_t states_ = 0;
  AscendC::GlobalTensor<uint32_t> positions_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> position_buffer_;
};
}  // namespace

extern "C" __global__ __aicore__ void statecentric_leapquant_window_boundary(
    GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
    GM_ADDR compensator_keys, GM_ADDR compensator_values,
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
    GM_ADDR positions, GM_ADDR next_residual, GM_ADDR next_smoothing,
    GM_ADDR next_scales, GM_ADDR next_compensator_keys,
    GM_ADDR next_compensator_values, GM_ADDR residual_workspace,
    uint32_t states,
    uint32_t power_iterations, uint32_t record_merge_ready) {
  WindowBoundaryKernel kernel;
  kernel.Init(residual, smoothing, scales, compensator_keys,
              compensator_values, record_decay, record_keys,
              record_corrections, positions, next_residual, next_smoothing,
              next_scales, next_compensator_keys, next_compensator_values,
              residual_workspace, states, power_iterations,
              record_merge_ready);
  kernel.Process();
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_boundary_scale_records(
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR scaled_record_keys,
    uint32_t states) {
  BoundaryRecordScaleKernel kernel;
  kernel.Init(record_decay, record_keys, scaled_record_keys, states);
  kernel.Process();
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_window_boundary_finalize(GM_ADDR positions,
                                                 uint32_t states) {
  WindowBoundaryFinalizeKernel kernel;
  kernel.Init(positions, states);
  kernel.Process();
}

extern "C" int32_t statecentric_leapquant_window_boundary_launch_v1(
    void* stream, const int8_t* residual, const float* smoothing,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, int8_t* next_residual, float* next_smoothing,
    float* next_scales, uint16_t* next_compensator_keys,
    uint16_t* next_compensator_values, float* residual_workspace,
    uint32_t states,
    uint32_t power_iterations) {
  if (!stream || !residual || !smoothing || !scales || !compensator_keys ||
      !compensator_values || !record_decay || !record_keys ||
      !record_corrections || !positions || !next_residual ||
      !next_smoothing || !next_scales || !next_compensator_keys ||
      !next_compensator_values || !residual_workspace || states == 0 ||
      states > 16384 ||
      power_iterations > 32) {
    return 1;
  }
  statecentric_leapquant_window_boundary<<<64, nullptr, stream>>>(
      const_cast<int8_t*>(residual), const_cast<float*>(smoothing),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)),
      record_decay, reinterpret_cast<half*>(record_keys),
      reinterpret_cast<half*>(record_corrections), positions, next_residual,
      next_smoothing, next_scales,
      reinterpret_cast<half*>(next_compensator_keys),
      reinterpret_cast<half*>(next_compensator_values), residual_workspace,
      states,
      power_iterations, 0U);
  if (power_iterations == 0) {
    statecentric_leapquant_window_boundary_finalize<<<64, nullptr, stream>>>(
        positions, states);
  }
  return 0;
}

extern "C" int32_t
statecentric_leapquant_boundary_scale_records_launch_v1(
    void* stream, const float* record_decay, const uint16_t* record_keys,
    uint16_t* scaled_record_keys, uint32_t states) {
  if (!stream || !record_decay || !record_keys || !scaled_record_keys ||
      states == 0 || states > 128) {
    return 1;
  }
  statecentric_leapquant_boundary_scale_records<<<64, nullptr, stream>>>(
      const_cast<float*>(record_decay),
      reinterpret_cast<half*>(const_cast<uint16_t*>(record_keys)),
      reinterpret_cast<half*>(scaled_record_keys), states);
  return 0;
}

extern "C" int32_t
statecentric_leapquant_window_boundary_cube_finish_launch_v1(
    void* stream, const int8_t* residual, const float* smoothing,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, int8_t* next_residual, float* next_smoothing,
    float* next_scales, uint16_t* next_compensator_keys,
    uint16_t* next_compensator_values, float* record_merge,
    uint32_t states) {
  if (!stream || !residual || !smoothing || !scales || !compensator_keys ||
      !compensator_values || !record_decay || !record_keys ||
      !record_corrections || !positions || !next_residual ||
      !next_smoothing || !next_scales || !next_compensator_keys ||
      !next_compensator_values || !record_merge || states == 0 ||
      states > 128) {
    return 1;
  }
  statecentric_leapquant_window_boundary<<<64, nullptr, stream>>>(
      const_cast<int8_t*>(residual), const_cast<float*>(smoothing),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)),
      record_decay, reinterpret_cast<half*>(record_keys),
      reinterpret_cast<half*>(record_corrections), positions, next_residual,
      next_smoothing, next_scales,
      reinterpret_cast<half*>(next_compensator_keys),
      reinterpret_cast<half*>(next_compensator_values), record_merge, states,
      0U, 1U);
  statecentric_leapquant_window_boundary_finalize<<<64, nullptr, stream>>>(
      positions, states);
  return 0;
}

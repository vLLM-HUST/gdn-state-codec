#include "kernel_operator.h"

#include "statecentric/leapquant_windowed_gdn_kernel.h"

namespace {
constexpr uint32_t kDimension = 128;
constexpr uint32_t kElements = kDimension * kDimension;
constexpr uint32_t kRank = 4;
constexpr uint32_t kWindow = 16;
constexpr uint32_t kPositionStride = 8;
constexpr uint32_t kColumnShards = 2;
constexpr uint32_t kColumnsPerShard = kDimension / kColumnShards;
constexpr uint32_t kResidualTileColumns = 8;

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

class WindowedGdnStepKernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
      GM_ADDR compensator_keys, GM_ADDR compensator_values,
      GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
      GM_ADDR positions, GM_ADDR decay, GM_ADDR write_keys, GM_ADDR reads,
      GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
    states_ = states;
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(residual), states * kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing), states * kDimension);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales), states * kDimension);
    compensator_keys_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(compensator_keys), states * kDimension * kRank);
    compensator_values_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(compensator_values), states * kDimension * kRank);
    record_decay_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(record_decay), states * kWindow);
    record_keys_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_keys), states * kWindow * kDimension);
    record_corrections_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_corrections), states * kWindow * kDimension);
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(positions),
                               states * kPositionStride);
    decay_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay), states);
    write_keys_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(write_keys), states * kDimension);
    reads_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(reads), states * kDimension);
    queries_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queries), states * kDimension);
    values_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(values), states * kDimension);
    outputs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputs), states * kDimension);
    pipe_.InitBuffer(position_buffer_, 32);
  }

  __aicore__ inline float ReadBoundaryElement(uint32_t state, uint32_t column,
                                              bool output_read,
                                              float boundary_suffix) {
    const uint32_t vector_base = state * kDimension;
    float residual_projection = 0.0F;
    const uint32_t boundary_base = state * kElements;
    for (uint32_t row = 0; row < kDimension; ++row) {
      const float x = output_read
                          ? static_cast<float>(queries_.GetValue(vector_base + row)) * decay_.GetValue(state)
                          : static_cast<float>(reads_.GetValue(vector_base + row));
      residual_projection += static_cast<float>(residual_.GetValue(boundary_base + row * kDimension + column)) *
                             smoothing_.GetValue(vector_base + row) *
                             boundary_suffix * x;
    }
    return residual_projection / 127.0F *
           scales_.GetValue(vector_base + column);
  }

  __aicore__ inline void Process() {
    for (uint32_t state = AscendC::GetBlockIdx(); state < states_;
         state += AscendC::GetBlockNum()) {
      const uint32_t position =
          positions_.GetValue(state * kPositionStride);
      if (position >= kWindow) continue;
      const uint32_t vector_base = state * kDimension;
      const uint32_t record_base = (state * kWindow + position) * kDimension;
      const uint32_t compensator_base = state * kDimension * kRank;
      float record_read_projection[kWindow];
      float record_output_projection[kWindow];
      float record_suffix[kWindow];
      float compensator_read_projection[kRank] = {0.0F, 0.0F, 0.0F, 0.0F};
      float compensator_output_projection[kRank] = {0.0F, 0.0F, 0.0F, 0.0F};
      const float current_decay = decay_.GetValue(state);
      for (uint32_t record = 0; record < position; ++record) {
        record_read_projection[record] = 0.0F;
        record_output_projection[record] = 0.0F;
        const uint32_t old_record_base =
            (state * kWindow + record) * kDimension;
        for (uint32_t row = 0; row < kDimension; ++row) {
          const float key = static_cast<float>(
              record_keys_.GetValue(old_record_base + row));
          record_read_projection[record] +=
              key * static_cast<float>(reads_.GetValue(vector_base + row));
          record_output_projection[record] +=
              key * static_cast<float>(queries_.GetValue(vector_base + row)) *
              current_decay;
        }
      }
      for (uint32_t rank = 0; rank < kRank; ++rank) {
        for (uint32_t row = 0; row < kDimension; ++row) {
          const float key = static_cast<float>(compensator_keys_.GetValue(
              compensator_base + row * kRank + rank));
          compensator_read_projection[rank] +=
              key * static_cast<float>(reads_.GetValue(vector_base + row));
          compensator_output_projection[rank] +=
              key * static_cast<float>(queries_.GetValue(vector_base + row)) *
              current_decay;
        }
      }
      float suffix = 1.0F;
      for (uint32_t reverse = position; reverse > 0; --reverse) {
        const uint32_t record = reverse - 1;
        record_suffix[record] = suffix;
        suffix *= record_decay_.GetValue(state * kWindow + record);
      }
      const float boundary_suffix = suffix;
      float update_projection = 0.0F;
      for (uint32_t row = 0; row < kDimension; ++row) {
        update_projection += static_cast<float>(write_keys_.GetValue(vector_base + row)) *
                             static_cast<float>(queries_.GetValue(vector_base + row));
      }
      for (uint32_t column = 0; column < kDimension; ++column) {
        float state_read = ReadBoundaryElement(
            state, column, false, boundary_suffix);
        float output_read = ReadBoundaryElement(
            state, column, true, boundary_suffix);
        for (uint32_t rank = 0; rank < kRank; ++rank) {
          const float compensator_value = static_cast<float>(
              compensator_values_.GetValue(
                  compensator_base + column * kRank + rank));
          state_read += compensator_value *
                        compensator_read_projection[rank] * boundary_suffix;
          output_read += compensator_value *
                         compensator_output_projection[rank] *
                         boundary_suffix;
        }
        for (uint32_t record = 0; record < position; ++record) {
          const float correction_value = static_cast<float>(
              record_corrections_.GetValue(
                  (state * kWindow + record) * kDimension + column));
          state_read += correction_value * record_read_projection[record] *
                        record_suffix[record];
          output_read += correction_value *
                         record_output_projection[record] *
                         record_suffix[record];
        }
        const float correction =
            static_cast<float>(values_.GetValue(vector_base + column)) -
            state_read;
        outputs_.SetValue(vector_base + column,
                          output_read + correction * update_projection);
        record_keys_.SetValue(record_base + column, write_keys_.GetValue(vector_base + column));
        record_corrections_.SetValue(record_base + column, static_cast<half>(correction));
      }
      record_decay_.SetValue(state * kWindow + position, decay_.GetValue(state));
      auto position_output = position_buffer_.Get<uint32_t>();
      position_output.SetValue(0, position + 1);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      const AscendC::DataCopyExtParams position_copy{
          1, sizeof(uint32_t), 0, 0, 0};
      AscendC::DataCopyPad(positions_[state * kPositionStride],
                          position_output, position_copy);
    }
  }

 private:
  uint32_t states_ = 0;
  AscendC::GlobalTensor<int8_t> residual_;
  AscendC::GlobalTensor<float> smoothing_, scales_, record_decay_, decay_, outputs_;
  AscendC::GlobalTensor<half> compensator_keys_, compensator_values_, record_keys_, record_corrections_;
  AscendC::GlobalTensor<half> write_keys_, reads_, queries_, values_;
  AscendC::GlobalTensor<uint32_t> positions_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> position_buffer_;
};

class WindowedGdnStepBf16Kernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
      GM_ADDR compensator_keys, GM_ADDR compensator_values,
      GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
      GM_ADDR positions, GM_ADDR decay, GM_ADDR beta, GM_ADDR keys,
      GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
    states_ = states;
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(residual),
                              states * kElements);
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing),
                               states * kDimension);
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales),
                            states * kDimension);
    compensator_keys_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(compensator_keys),
        states * kDimension * kRank);
    compensator_values_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(compensator_values),
        states * kDimension * kRank);
    record_decay_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(record_decay),
                                  states * kWindow);
    record_keys_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(record_keys),
                                 states * kWindow * kDimension);
    record_corrections_.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(record_corrections),
        states * kWindow * kDimension);
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(positions),
                               states * kPositionStride);
    decay_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay), states);
    beta_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(beta), states);
    keys_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(keys),
                          states / 2 * kDimension);
    queries_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(queries),
                             states / 2 * kDimension);
    values_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(values),
                            states * kDimension);
    outputs_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(outputs),
                             states * kDimension);
    pipe_.InitBuffer(position_buffer_, 32);
    pipe_.InitBuffer(decay_buffer_, 32);
    pipe_.InitBuffer(residual_int8_buffer_,
                     kResidualTileColumns * kDimension * sizeof(int8_t));
    pipe_.InitBuffer(residual_half_buffer_,
                     kResidualTileColumns * kDimension * sizeof(half));
    pipe_.InitBuffer(residual_float_buffer_,
                     2 * kResidualTileColumns * kDimension * sizeof(half));
    pipe_.InitBuffer(read_input_buffer_, 2 * kDimension * sizeof(half));
    pipe_.InitBuffer(key_query_buffer_, 2 * kDimension * sizeof(uint16_t));
    pipe_.InitBuffer(smoothing_local_buffer_, kDimension * sizeof(float));
    pipe_.InitBuffer(compensator_key_local_buffer_,
                     kDimension * kRank * sizeof(half));
    pipe_.InitBuffer(compensator_value_local_buffer_,
                     kColumnsPerShard * kRank * sizeof(half));
    pipe_.InitBuffer(value_local_buffer_,
                     kColumnsPerShard * sizeof(uint16_t));
    pipe_.InitBuffer(projection_vector_buffer_,
                     2 * kDimension * sizeof(float));
    pipe_.InitBuffer(product_buffer_, 2 * kDimension * sizeof(float));
    pipe_.InitBuffer(reduction_buffer_, 32);
    pipe_.InitBuffer(record_half_buffer_,
                     kWindow * kDimension * sizeof(half));
    pipe_.InitBuffer(record_float_buffer_,
                     kWindow * kDimension * sizeof(float));
    pipe_.InitBuffer(correction_half_buffer_,
                     kWindow * kDimension * sizeof(half));
    pipe_.InitBuffer(state_read_buffer_,
                     (kDimension / 2) * sizeof(half));
    pipe_.InitBuffer(output_read_buffer_,
                     (kDimension / 2) * sizeof(half));
  }

  __aicore__ inline void RecordProjectionPair(
      uint32_t record,
      const AscendC::LocalTensor<float>& record_keys,
      const AscendC::LocalTensor<float>& projection_vectors,
      float& state_read, float& output_read) {
    auto basis_float = record_keys[record * kDimension];
    auto product = product_buffer_.Get<float>();
    auto reduction = reduction_buffer_.Get<float>();
    AscendC::Mul(product, basis_float, projection_vectors, kDimension);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(product[128], basis_float,
                 projection_vectors[kDimension], kDimension);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(product, product, product[64], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(product[64], product[128], product[192], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(reduction, product, 64, 2, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    state_read = reduction.GetValue(0);
    output_read = reduction.GetValue(1);
  }

  __aicore__ inline void BoundaryProjectionTile(
      uint32_t state, uint32_t tile_begin, uint32_t column_begin,
      const AscendC::LocalTensor<half>& projection_inputs,
      float boundary_suffix,
      const AscendC::LocalTensor<half>& state_reads,
      const AscendC::LocalTensor<half>& output_reads) {
    constexpr uint32_t tile_elements =
        kResidualTileColumns * kDimension;
    auto residual_half = residual_half_buffer_.Get<half>();
    auto product = residual_float_buffer_.Get<half>();
    auto reduction = reduction_buffer_.Get<half>();
    AscendC::Mul(product, residual_half, projection_inputs,
                 kDimension, kResidualTileColumns,
                 {1, 1, 1, 8, 8, 0});
    AscendC::Mul(product[tile_elements], residual_half,
                 projection_inputs[kDimension],
                 kDimension, kResidualTileColumns,
                 {1, 1, 1, 8, 8, 0});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(
        reduction, product, kDimension,
        2 * kResidualTileColumns, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t tile_column = 0;
         tile_column < kResidualTileColumns; ++tile_column) {
      const uint32_t column = tile_begin + tile_column;
      const float output_scale =
          scales_.GetValue(state * kDimension + column) *
          boundary_suffix / 127.0F;
      state_reads.SetValue(
          column - column_begin,
          static_cast<half>(
              static_cast<float>(reduction.GetValue(tile_column)) *
              output_scale));
      output_reads.SetValue(
          column - column_begin,
          static_cast<half>(static_cast<float>(reduction.GetValue(
              kResidualTileColumns + tile_column)) * output_scale));
    }
  }

  template <bool UseCompensator>
  __aicore__ inline void Process() {
    constexpr float query_scale = 0.08838834764831845F;
    for (uint32_t task = AscendC::GetBlockIdx();
         task < states_ * kColumnShards;
         task += AscendC::GetBlockNum()) {
      const uint32_t state = task / kColumnShards;
      const uint32_t column_shard = task % kColumnShards;
      const uint32_t column_begin = column_shard * kColumnsPerShard;
      const uint32_t column_end = column_begin + kColumnsPerShard;
      const uint32_t position =
          positions_.GetValue(state * kPositionStride);
      if (position >= kWindow) continue;
      const uint32_t vector_base = state * kDimension;
      const uint32_t query_base = (state / 2) * kDimension;
      const uint32_t record_base =
          (state * kWindow + position) * kDimension;
      const uint32_t compensator_base = state * kDimension * kRank;
      auto decay_local = decay_buffer_.Get<float>();
      decay_local.SetValue(0, decay_.GetValue(state));
      AscendC::Exp(decay_local, decay_local, 1);
      AscendC::PipeBarrier<PIPE_V>();
      const float current_decay = decay_local.GetValue(0);
      const float current_beta = BFloat16ToFloat(beta_.GetValue(state));
      auto read_input = read_input_buffer_.Get<half>();
      auto output_input = read_input[kDimension];
      auto key_local = key_query_buffer_.Get<uint16_t>();
      auto query_local = key_local[kDimension];
      auto smoothing_local = smoothing_local_buffer_.Get<float>();
      auto compensator_key_local =
          compensator_key_local_buffer_.Get<half>();
      auto compensator_value_local =
          compensator_value_local_buffer_.Get<half>();
      auto value_local = value_local_buffer_.Get<uint16_t>();
      auto read_vector = projection_vector_buffer_.Get<float>();
      auto output_vector = read_vector[kDimension];
      auto record_half = record_half_buffer_.Get<half>();
      auto record_float = record_float_buffer_.Get<float>();
      auto correction_half = correction_half_buffer_.Get<half>();
      auto state_reads = state_read_buffer_.Get<half>();
      auto output_reads = output_read_buffer_.Get<half>();
      float record_read_projection[kWindow] = {};
      float record_output_projection[kWindow] = {};
      float record_suffix[kWindow];
      float compensator_read_projection[kRank] = {};
      float compensator_output_projection[kRank] = {};
      float update_projection = 0.0F;
      AscendC::DataCopy(key_local, keys_[query_base], kDimension);
      AscendC::DataCopy(query_local, queries_[query_base], kDimension);
      AscendC::DataCopy(smoothing_local, smoothing_[vector_base], kDimension);
      if constexpr (UseCompensator) {
        AscendC::DataCopy(compensator_key_local,
                          compensator_keys_[compensator_base],
                          kDimension * kRank);
        AscendC::DataCopy(
            compensator_value_local,
            compensator_values_[compensator_base + column_begin * kRank],
            kColumnsPerShard * kRank);
      }
      AscendC::DataCopy(value_local,
                        values_[vector_base + column_begin],
                        kColumnsPerShard);
      if constexpr (UseCompensator) {
        const event_t metadata_ready = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(metadata_ready);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(metadata_ready);
        for (uint32_t row = 0; row < kDimension; ++row) {
          const float key = BFloat16ToFloat(key_local.GetValue(row));
          const float query = query_scale *
              BFloat16ToFloat(query_local.GetValue(row));
          const float read = current_decay * key;
          const float write_key = current_beta * key;
          const float smoothing = smoothing_local.GetValue(row);
          read_vector.SetValue(row, read);
          output_vector.SetValue(row, query * current_decay);
          read_input.SetValue(row, static_cast<half>(read * smoothing));
          output_input.SetValue(
              row, static_cast<half>(query * current_decay * smoothing));
          update_projection += write_key * query;
          for (uint32_t rank = 0; rank < kRank; ++rank) {
            const float basis = static_cast<float>(
                compensator_key_local.GetValue(row * kRank + rank));
            compensator_read_projection[rank] += basis * read;
            compensator_output_projection[rank] +=
                basis * query * current_decay;
          }
        }
      } else {
        const event_t metadata_ready = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metadata_ready);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metadata_ready);
        auto key_bf16 = key_local.ReinterpretCast<bfloat16_t>();
        auto query_bf16 = query_local.ReinterpretCast<bfloat16_t>();
        auto product = product_buffer_.Get<float>();
        auto reduction = reduction_buffer_.Get<float>();
        AscendC::Cast(read_vector, key_bf16,
                      AscendC::RoundMode::CAST_NONE, kDimension);
        AscendC::Cast(output_vector, query_bf16,
                      AscendC::RoundMode::CAST_NONE, kDimension);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, read_vector, output_vector, kDimension);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(product, product, product[64], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(reduction, product, 64, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        const event_t vector_to_scalar = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
        update_projection =
            reduction.GetValue(0) * current_beta * query_scale;
        const event_t scalar_to_vector = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
        AscendC::Muls(read_vector, read_vector, current_decay, kDimension);
        AscendC::Muls(output_vector, output_vector,
                      current_decay * query_scale, kDimension);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, read_vector, smoothing_local, kDimension);
        AscendC::Mul(product[kDimension], output_vector, smoothing_local,
                     kDimension);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(read_input, product,
                      AscendC::RoundMode::CAST_NONE, kDimension);
        AscendC::Cast(output_input, product[kDimension],
                      AscendC::RoundMode::CAST_NONE, kDimension);
      }
      AscendC::PipeBarrier<PIPE_V>();
      if (position > 0) {
        AscendC::DataCopy(record_half,
                          record_keys_[state * kWindow * kDimension],
                          position * kDimension);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(record_float, record_half,
                      AscendC::RoundMode::CAST_NONE,
                      position * kDimension);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(
            correction_half,
            record_corrections_[state * kWindow * kDimension],
            position * kDimension);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      }
      for (uint32_t record = 0; record < position; ++record) {
        RecordProjectionPair(
            record, record_float, read_vector,
            record_read_projection[record],
            record_output_projection[record]);
      }
      float suffix = 1.0F;
      for (uint32_t reverse = position; reverse > 0; --reverse) {
        const uint32_t record = reverse - 1;
        record_suffix[record] = suffix;
        suffix *= record_decay_.GetValue(state * kWindow + record);
      }
      auto residual_int8 = residual_int8_buffer_.Get<int8_t>();
      auto residual_half = residual_half_buffer_.Get<half>();
      const uint32_t boundary_base = state * kElements;
      constexpr uint32_t residual_tile_elements =
          kResidualTileColumns * kDimension;
      for (uint32_t tile_begin = column_begin; tile_begin < column_end;
           tile_begin += kResidualTileColumns) {
        AscendC::DataCopy(
            residual_int8,
            residual_[boundary_base + tile_begin * kDimension],
            residual_tile_elements);
        const event_t residual_ready = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(residual_ready);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(residual_ready);
        AscendC::Cast(residual_half, residual_int8,
                      AscendC::RoundMode::CAST_NONE,
                      residual_tile_elements);
        AscendC::PipeBarrier<PIPE_V>();
        BoundaryProjectionTile(
            state, tile_begin, column_begin, read_input, suffix,
            state_reads, output_reads);
        if constexpr (UseCompensator) {
          for (uint32_t column = tile_begin;
               column < tile_begin + kResidualTileColumns; ++column) {
            float state_read = static_cast<float>(
                state_reads.GetValue(column - column_begin));
            float output_read = static_cast<float>(
                output_reads.GetValue(column - column_begin));
            for (uint32_t rank = 0; rank < kRank; ++rank) {
              const float basis = static_cast<float>(
                  compensator_value_local.GetValue(
                      (column - column_begin) * kRank + rank));
              state_read +=
                  basis * compensator_read_projection[rank] * suffix;
              output_read +=
                  basis * compensator_output_projection[rank] * suffix;
            }
            state_reads.SetValue(column - column_begin,
                                 static_cast<half>(state_read));
            output_reads.SetValue(column - column_begin,
                                  static_cast<half>(output_read));
          }
        }
      }
      const event_t scalar_to_vector = static_cast<event_t>(
          pipe_.FetchEventID(AscendC::HardEvent::S_V));
      AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
      for (uint32_t record = 0; record < position; ++record) {
        auto correction = correction_half[
            record * kDimension + column_begin];
        AscendC::Axpy(state_reads, correction,
                      static_cast<half>(record_read_projection[record] *
                                        record_suffix[record]),
                      kColumnsPerShard);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Axpy(output_reads, correction,
                      static_cast<half>(record_output_projection[record] *
                                        record_suffix[record]),
                      kColumnsPerShard);
        AscendC::PipeBarrier<PIPE_V>();
      }
      const event_t vector_to_scalar = static_cast<event_t>(
          pipe_.FetchEventID(AscendC::HardEvent::V_S));
      AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
      // Finish the shard as vectors.  The previous scalar loop performed 64
      // BF16/FP16 conversions and three scalar GM stores per task, making the
      // quant-only decode path scalar-bound.  All source tensors are already
      // contiguous in UB, so form the correction, readout, and record payloads
      // in FP32 and issue one MTE3 copy for each result.
      auto final_float = product_buffer_.Get<float>();
      auto value_float = final_float;
      auto state_float = final_float[kColumnsPerShard];
      auto output_float = final_float[2 * kColumnsPerShard];
      auto key_float = final_float[3 * kColumnsPerShard];
      auto value_bf16 = value_local.ReinterpretCast<bfloat16_t>();
      auto key_bf16 = key_local.ReinterpretCast<bfloat16_t>();
      auto output_bf16 = value_bf16;
      auto output_u16 = value_local;
      auto key_half = read_input;
      auto correction_output_half = state_reads;
      AscendC::Cast(value_float, value_bf16,
                    AscendC::RoundMode::CAST_NONE, kColumnsPerShard);
      AscendC::Cast(state_float, state_reads,
                    AscendC::RoundMode::CAST_NONE, kColumnsPerShard);
      AscendC::Cast(output_float, output_reads,
                    AscendC::RoundMode::CAST_NONE, kColumnsPerShard);
      AscendC::Cast(key_float, key_bf16[column_begin],
                    AscendC::RoundMode::CAST_NONE, kColumnsPerShard);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(value_float, value_float, state_float,
                   kColumnsPerShard);
      AscendC::Muls(key_float, key_float, current_beta,
                    kColumnsPerShard);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(state_float, value_float, update_projection,
                    kColumnsPerShard);
      AscendC::Add(output_float, output_float, state_float,
                   kColumnsPerShard);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(output_bf16, output_float,
                    AscendC::RoundMode::CAST_RINT, kColumnsPerShard);
      AscendC::Cast(key_half, key_float,
                    AscendC::RoundMode::CAST_NONE, kColumnsPerShard);
      AscendC::Cast(correction_output_half, value_float,
                    AscendC::RoundMode::CAST_NONE, kColumnsPerShard);
      AscendC::PipeBarrier<PIPE_V>();
      const event_t results_ready = static_cast<event_t>(
          pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(results_ready);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(results_ready);
      AscendC::DataCopy(outputs_[vector_base + column_begin], output_u16,
                        kColumnsPerShard);
      AscendC::DataCopy(record_keys_[record_base + column_begin], key_half,
                        kColumnsPerShard);
      AscendC::DataCopy(
          record_corrections_[record_base + column_begin],
          correction_output_half, kColumnsPerShard);
      if (column_shard == 0) {
        record_decay_.SetValue(state * kWindow + position, current_decay);
        auto position_output = position_buffer_.Get<uint32_t>();
        position_output.SetValue(0, position + 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const AscendC::DataCopyExtParams position_copy{
            1, sizeof(uint32_t), 0, 0, 0};
        AscendC::DataCopyPad(positions_[state * kPositionStride],
                            position_output, position_copy);
      }
    }
  }

 private:
  uint32_t states_ = 0;
  AscendC::GlobalTensor<int8_t> residual_;
  AscendC::GlobalTensor<float> smoothing_, scales_, record_decay_, decay_;
  AscendC::GlobalTensor<half> compensator_keys_, compensator_values_;
  AscendC::GlobalTensor<half> record_keys_, record_corrections_;
  AscendC::GlobalTensor<uint16_t> beta_, keys_, queries_, values_, outputs_;
  AscendC::GlobalTensor<uint32_t> positions_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> position_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> decay_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> residual_int8_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> residual_half_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> residual_float_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> read_input_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> key_query_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> smoothing_local_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC>
      compensator_key_local_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC>
      compensator_value_local_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> value_local_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> projection_vector_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> product_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> reduction_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> record_half_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> record_float_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> correction_half_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> state_read_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> output_read_buffer_;
};
}  // namespace

extern "C" __global__ __aicore__ void statecentric_leapquant_windowed_gdn_step(
    GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
    GM_ADDR compensator_keys, GM_ADDR compensator_values,
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
    GM_ADDR positions, GM_ADDR decay, GM_ADDR write_keys, GM_ADDR reads,
    GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
  WindowedGdnStepKernel kernel;
  kernel.Init(residual, smoothing, scales, compensator_keys, compensator_values,
              record_decay, record_keys, record_corrections, positions, decay,
              write_keys, reads, queries, values, outputs, states);
  kernel.Process();
}

extern "C" int32_t statecentric_leapquant_windowed_gdn_step_launch_v1(
    void* stream, const int8_t* residual, const float* smoothing,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, const float* decay, const uint16_t* write_keys,
    const uint16_t* reads, const uint16_t* queries, const uint16_t* values,
    float* outputs, uint32_t states) {
  if (!stream || !residual || !smoothing || !scales || !compensator_keys ||
      !compensator_values || !record_decay || !record_keys ||
      !record_corrections || !positions || !decay || !write_keys || !reads ||
      !queries || !values || !outputs || states == 0 || states > 16384) return 1;
  statecentric_leapquant_windowed_gdn_step<<<64, nullptr, stream>>>(
      const_cast<int8_t*>(residual), const_cast<float*>(smoothing),
      const_cast<float*>(scales), reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)), record_decay,
      reinterpret_cast<half*>(record_keys), reinterpret_cast<half*>(record_corrections),
      positions, const_cast<float*>(decay), reinterpret_cast<half*>(const_cast<uint16_t*>(write_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(reads)), reinterpret_cast<half*>(const_cast<uint16_t*>(queries)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(values)), outputs, states);
  return 0;
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_windowed_gdn_step_bf16(
    GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
    GM_ADDR compensator_keys, GM_ADDR compensator_values,
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
    GM_ADDR positions, GM_ADDR decay, GM_ADDR beta, GM_ADDR keys,
    GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
  WindowedGdnStepBf16Kernel kernel;
  kernel.Init(residual, smoothing, scales, compensator_keys,
              compensator_values, record_decay, record_keys,
              record_corrections, positions, decay, beta, keys, queries,
              values, outputs, states);
  kernel.Process<true>();
}

extern "C" __global__ __aicore__ void
statecentric_leapquant_windowed_gdn_step_bf16_quant_only(
    GM_ADDR residual, GM_ADDR smoothing, GM_ADDR scales,
    GM_ADDR compensator_keys, GM_ADDR compensator_values,
    GM_ADDR record_decay, GM_ADDR record_keys, GM_ADDR record_corrections,
    GM_ADDR positions, GM_ADDR decay, GM_ADDR beta, GM_ADDR keys,
    GM_ADDR queries, GM_ADDR values, GM_ADDR outputs, uint32_t states) {
  WindowedGdnStepBf16Kernel kernel;
  kernel.Init(residual, smoothing, scales, compensator_keys,
              compensator_values, record_decay, record_keys,
              record_corrections, positions, decay, beta, keys, queries,
              values, outputs, states);
  kernel.Process<false>();
}

extern "C" int32_t
statecentric_leapquant_windowed_gdn_step_bf16_launch_v1(
    void* stream, const int8_t* residual, const float* smoothing,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, const float* decay, const uint16_t* beta,
    const uint16_t* keys, const uint16_t* queries, const uint16_t* values,
    uint16_t* outputs, uint32_t states) {
  if (!stream || !residual || !smoothing || !scales || !compensator_keys ||
      !compensator_values || !record_decay || !record_keys ||
      !record_corrections || !positions || !decay || !beta || !keys ||
      !queries || !values || !outputs || states == 0 || states > 16384 ||
      states % 2 != 0)
    return 1;
  statecentric_leapquant_windowed_gdn_step_bf16<<<64, nullptr, stream>>>(
      const_cast<int8_t*>(residual), const_cast<float*>(smoothing),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)),
      record_decay, reinterpret_cast<half*>(record_keys),
      reinterpret_cast<half*>(record_corrections), positions,
      const_cast<float*>(decay), const_cast<uint16_t*>(beta),
      const_cast<uint16_t*>(keys), const_cast<uint16_t*>(queries),
      const_cast<uint16_t*>(values), outputs, states);
  return 0;
}

extern "C" int32_t
statecentric_leapquant_windowed_gdn_step_bf16_quant_only_launch_v1(
    void* stream, const int8_t* residual, const float* smoothing,
    const float* scales, const uint16_t* compensator_keys,
    const uint16_t* compensator_values, float* record_decay,
    uint16_t* record_keys, uint16_t* record_corrections,
    uint32_t* positions, const float* decay, const uint16_t* beta,
    const uint16_t* keys, const uint16_t* queries, const uint16_t* values,
    uint16_t* outputs, uint32_t states) {
  if (!stream || !residual || !smoothing || !scales || !compensator_keys ||
      !compensator_values || !record_decay || !record_keys ||
      !record_corrections || !positions || !decay || !beta || !keys ||
      !queries || !values || !outputs || states == 0 || states > 16384 ||
      states % 2 != 0)
    return 1;
  statecentric_leapquant_windowed_gdn_step_bf16_quant_only
      <<<states * kColumnShards, nullptr, stream>>>(
      const_cast<int8_t*>(residual), const_cast<float*>(smoothing),
      const_cast<float*>(scales),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_keys)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(compensator_values)),
      record_decay, reinterpret_cast<half*>(record_keys),
      reinterpret_cast<half*>(record_corrections), positions,
      const_cast<float*>(decay), const_cast<uint16_t*>(beta),
      const_cast<uint16_t*>(keys), const_cast<uint16_t*>(queries),
      const_cast<uint16_t*>(values), outputs, states);
  return 0;
}

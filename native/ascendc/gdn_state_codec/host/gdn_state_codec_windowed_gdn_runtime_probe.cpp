#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "statecentric/gdn_state_codec_windowed_gdn_kernel.h"

namespace {

constexpr std::size_t kD = 128;
constexpr std::size_t kElements = kD * kD;
constexpr std::size_t kRank = 4;
constexpr std::size_t kWindow = 16;
constexpr std::size_t kPositionStride = 8;

void Check(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(status));
  }
}

std::uint16_t Half(float value) {
  return static_cast<std::uint16_t>(aclFloatToFloat16(value));
}

float Float(std::uint16_t value) {
  return aclFloat16ToFloat(static_cast<aclFloat16>(value));
}

template <class Type>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    Check(aclrtMalloc(reinterpret_cast<void**>(&data_), bytes(),
                      ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)aclrtFree(data_);
  }
  Type* data() const { return data_; }
  std::size_t bytes() const { return elements_ * sizeof(Type); }

 private:
  Type* data_ = nullptr;
  std::size_t elements_ = 0;
};

template <class Type>
void ToDevice(DeviceBuffer<Type>& destination,
              const std::vector<Type>& source) {
  if (destination.bytes() != source.size() * sizeof(Type)) {
    throw std::invalid_argument("host/device buffer size mismatch");
  }
  Check(aclrtMemcpy(destination.data(), destination.bytes(), source.data(),
                    destination.bytes(), ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy H2D");
}

template <class Type>
std::vector<Type> FromDevice(const DeviceBuffer<Type>& source,
                             std::size_t elements) {
  std::vector<Type> output(elements);
  Check(aclrtMemcpy(output.data(), output.size() * sizeof(Type), source.data(),
                    source.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy D2H");
  return output;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <class Launch>
double Measure(aclrtStream stream, Launch&& launch) {
  Check(aclrtSynchronizeStream(stream), "sync before timed launch");
  const auto begin = std::chrono::steady_clock::now();
  launch();
  Check(aclrtSynchronizeStream(stream), "sync after timed launch");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

std::vector<float> Reconstruct(
    const std::vector<std::int8_t>& residual,
    const std::vector<float>& smoothing, const std::vector<float>& scales,
    const std::vector<std::uint16_t>& compensator_keys,
    const std::vector<std::uint16_t>& compensator_values,
    const std::vector<float>& record_decay,
    const std::vector<std::uint16_t>& record_keys,
    const std::vector<std::uint16_t>& record_corrections,
    const std::vector<std::uint32_t>& positions, std::size_t states) {
  std::vector<float> dense(states * kElements);
  for (std::size_t state = 0; state < states; ++state) {
    for (std::size_t row = 0; row < kD; ++row) {
      for (std::size_t column = 0; column < kD; ++column) {
        const std::size_t cell = state * kElements + row * kD + column;
        float value = static_cast<float>(residual[cell]) / 127.0F *
                      smoothing[state * kD + row] *
                      scales[state * kD + column];
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          value += Float(compensator_keys[
                       (state * kD + row) * kRank + rank]) *
                   Float(compensator_values[
                       (state * kD + column) * kRank + rank]);
        }
        dense[cell] = value;
      }
    }
    for (std::size_t record = 0;
         record < positions[state * kPositionStride]; ++record) {
      const std::size_t record_base = (state * kWindow + record) * kD;
      for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t column = 0; column < kD; ++column) {
          const std::size_t cell = state * kElements + row * kD + column;
          dense[cell] = record_decay[state * kWindow + record] * dense[cell] +
                        Float(record_keys[record_base + row]) *
                            Float(record_corrections[record_base + column]);
        }
      }
    }
  }
  return dense;
}

double RelativeRms(const std::vector<float>& actual,
                   const std::vector<float>& reference) {
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference = actual[index] - reference[index];
    squared_error += difference * difference;
    squared_reference += static_cast<double>(reference[index]) *
                         reference[index];
  }
  return std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
}

}  // namespace

int main(int argc, char** argv) {
  bool initialized = false;
  bool device_set = false;
  int device = -1;
  aclrtStream stream = nullptr;
  try {
    if (argc < 2 || argc > 4) {
      throw std::invalid_argument("usage: probe DEVICE [POSITION] [BATCH]");
    }
    device = std::stoi(argv[1]);
    const std::uint32_t initial_position =
        argc >= 3 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 7U;
    if (initial_position >= kWindow) {
      throw std::invalid_argument("POSITION must be in [0,15]");
    }
    const std::size_t batch =
        argc == 4 ? static_cast<std::size_t>(std::stoul(argv[3])) : 1U;
    if (batch == 0 || batch > 512) {
      throw std::invalid_argument("BATCH must be in [1,512]");
    }
    const std::size_t states = batch * 32U;
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<std::int8_t> residual(states * kElements);
    std::vector<float> smoothing(states * kD);
    std::vector<float> scales(states * kD);
    std::vector<std::uint16_t> compensator_keys(states * kD * kRank);
    std::vector<std::uint16_t> compensator_values(states * kD * kRank);
    std::vector<float> record_decay(states * kWindow, 0.0F);
    std::vector<std::uint16_t> record_keys(states * kWindow * kD, Half(0));
    std::vector<std::uint16_t> record_corrections(states * kWindow * kD,
                                                  Half(0));
    std::vector<std::uint32_t> positions(states * kPositionStride, 0);
    std::vector<float> decay(states);
    std::vector<std::uint16_t> write_keys(states * kD);
    std::vector<std::uint16_t> reads(states * kD);
    std::vector<std::uint16_t> queries(states * kD);
    std::vector<std::uint16_t> values(states * kD);

    for (std::size_t state = 0; state < states; ++state) {
      positions[state * kPositionStride] = initial_position;
      decay[state] = 0.94F + 0.001F * static_cast<float>(state % 7);
      for (std::size_t row = 0; row < kD; ++row) {
        smoothing[state * kD + row] =
            0.75F + 0.002F * static_cast<float>((state + row) % 41);
        scales[state * kD + row] =
            0.11F + 0.001F * static_cast<float>((3 * state + row) % 29);
        write_keys[state * kD + row] = Half(
            0.035F * std::sin(static_cast<float>(state * kD + row + 1) *
                             0.013F));
        reads[state * kD + row] = Half(
            0.031F * std::cos(static_cast<float>(state * kD + row + 3) *
                             0.017F));
        queries[state * kD + row] = Half(
            0.029F * std::sin(static_cast<float>(state * kD + row + 5) *
                             0.019F));
        values[state * kD + row] = Half(
            0.12F * std::cos(static_cast<float>(state * kD + row + 7) *
                            0.011F));
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          compensator_keys[(state * kD + row) * kRank + rank] = Half(
              0.008F * std::sin(static_cast<float>(state + row + rank + 1) *
                               0.071F));
          compensator_values[(state * kD + row) * kRank + rank] = Half(
              0.009F * std::cos(static_cast<float>(state + row + rank + 2) *
                               0.067F));
        }
      }
      for (std::size_t index = 0; index < kElements; ++index) {
        residual[state * kElements + index] = static_cast<std::int8_t>(
            static_cast<int>((index * 17 + state * 11) % 255) - 127);
      }
      for (std::size_t record = 0; record < initial_position; ++record) {
        record_decay[state * kWindow + record] =
            0.93F + 0.002F * static_cast<float>((state + record) % 9);
        const std::size_t base = (state * kWindow + record) * kD;
        for (std::size_t axis = 0; axis < kD; ++axis) {
          record_keys[base + axis] = Half(
              0.026F * std::sin(static_cast<float>(base + axis + 1) *
                               0.007F));
          record_corrections[base + axis] = Half(
              0.082F * std::cos(static_cast<float>(base + axis + 1) *
                               0.005F));
        }
      }
    }

    const auto initial_dense =
        Reconstruct(residual, smoothing, scales, compensator_keys,
                    compensator_values, record_decay, record_keys,
                    record_corrections, positions, states);

    DeviceBuffer<std::int8_t> residual_d(residual.size());
    DeviceBuffer<float> smoothing_d(smoothing.size());
    DeviceBuffer<float> scales_d(scales.size());
    DeviceBuffer<std::uint16_t> compensator_keys_d(compensator_keys.size());
    DeviceBuffer<std::uint16_t> compensator_values_d(compensator_values.size());
    DeviceBuffer<float> record_decay_d(record_decay.size());
    DeviceBuffer<std::uint16_t> record_keys_d(record_keys.size());
    DeviceBuffer<std::uint16_t> record_corrections_d(record_corrections.size());
    DeviceBuffer<std::uint32_t> positions_d(positions.size());
    DeviceBuffer<float> decay_d(decay.size());
    DeviceBuffer<std::uint16_t> write_keys_d(write_keys.size());
    DeviceBuffer<std::uint16_t> reads_d(reads.size());
    DeviceBuffer<std::uint16_t> queries_d(queries.size());
    DeviceBuffer<std::uint16_t> values_d(values.size());
    DeviceBuffer<float> window_outputs_d(states * kD);
    DeviceBuffer<float> dense_d(initial_dense.size());
    DeviceBuffer<float> fp32_outputs_d(states * kD);

    ToDevice(residual_d, residual);
    ToDevice(smoothing_d, smoothing);
    ToDevice(scales_d, scales);
    ToDevice(compensator_keys_d, compensator_keys);
    ToDevice(compensator_values_d, compensator_values);
    ToDevice(decay_d, decay);
    ToDevice(write_keys_d, write_keys);
    ToDevice(reads_d, reads);
    ToDevice(queries_d, queries);
    ToDevice(values_d, values);

    auto restore_window = [&] {
      ToDevice(record_decay_d, record_decay);
      ToDevice(record_keys_d, record_keys);
      ToDevice(record_corrections_d, record_corrections);
      ToDevice(positions_d, positions);
    };
    auto launch_window = [&] {
      if (statecentric_gdn_state_codec_windowed_gdn_step_launch_v1(
              stream, residual_d.data(), smoothing_d.data(), scales_d.data(),
              compensator_keys_d.data(), compensator_values_d.data(),
              record_decay_d.data(), record_keys_d.data(),
              record_corrections_d.data(), positions_d.data(), decay_d.data(),
              write_keys_d.data(), reads_d.data(), queries_d.data(),
              values_d.data(), window_outputs_d.data(), states) != 0) {
        throw std::runtime_error("windowed launch admission failed");
      }
    };
    auto restore_fp32 = [&] { ToDevice(dense_d, initial_dense); };
    auto launch_fp32 = [&] {
      if (statecentric_gdn_state_codec_fp32_gdn_step_launch_v1(
              stream, dense_d.data(), decay_d.data(), write_keys_d.data(),
              reads_d.data(), queries_d.data(), values_d.data(),
              fp32_outputs_d.data(), states) != 0) {
        throw std::runtime_error("FP32 launch admission failed");
      }
    };

    restore_window();
    restore_fp32();
    launch_window();
    launch_fp32();
    Check(aclrtSynchronizeStream(stream), "sync exactness");
    const auto window_outputs =
        FromDevice(window_outputs_d, states * kD);
    const auto fp32_outputs = FromDevice(fp32_outputs_d, states * kD);
    const auto fp32_dense = FromDevice(dense_d, initial_dense.size());
    const auto updated_record_decay =
        FromDevice(record_decay_d, record_decay.size());
    const auto updated_record_keys =
        FromDevice(record_keys_d, record_keys.size());
    const auto updated_record_corrections =
        FromDevice(record_corrections_d, record_corrections.size());
    const auto updated_positions =
        FromDevice(positions_d, positions.size());
    const auto window_dense = Reconstruct(
        residual, smoothing, scales, compensator_keys, compensator_values,
        updated_record_decay, updated_record_keys, updated_record_corrections,
        updated_positions, states);

    const double output_relative_rms =
        RelativeRms(window_outputs, fp32_outputs);
    const double state_relative_rms = RelativeRms(window_dense, fp32_dense);
    std::uint32_t minimum_position = kWindow;
    std::uint32_t maximum_position = 0;
    std::size_t invalid_positions = 0;
    for (std::size_t state = 0; state < states; ++state) {
      const std::uint32_t position =
          updated_positions[state * kPositionStride];
      minimum_position = std::min(minimum_position, position);
      maximum_position = std::max(maximum_position, position);
      invalid_positions += position == initial_position + 1 ? 0U : 1U;
    }
    const bool positions_valid = invalid_positions == 0;
    if (invalid_positions != 0) {
      std::cerr << "invalid position states:";
      for (std::size_t state = 0; state < states; ++state) {
        const std::uint32_t position =
            updated_positions[state * kPositionStride];
        if (position != initial_position + 1) {
          std::cerr << ' ' << state << '=' << position;
        }
      }
      std::cerr << '\n';
    }

    std::vector<double> window_us;
    std::vector<double> fp32_us;
    for (int warmup = 0; warmup < 3; ++warmup) {
      restore_window();
      launch_window();
      restore_fp32();
      launch_fp32();
    }
    Check(aclrtSynchronizeStream(stream), "sync warmup");
    for (int repeat = 0; repeat < 11; ++repeat) {
      if (repeat % 2 == 0) {
        restore_window();
        window_us.push_back(Measure(stream, launch_window));
        restore_fp32();
        fp32_us.push_back(Measure(stream, launch_fp32));
      } else {
        restore_fp32();
        fp32_us.push_back(Measure(stream, launch_fp32));
        restore_window();
        window_us.push_back(Measure(stream, launch_window));
      }
    }

    const bool pass = positions_valid && output_relative_rms < 0.0025 &&
                      state_relative_rms < 0.0025;
    std::cout << "{\"device\":" << device
              << ",\"batch\":" << batch << ",\"states\":" << states
              << ",\"heads_per_request\":32,\"position\":"
              << initial_position
              << ",\"output_relative_rms\":" << output_relative_rms
              << ",\"state_relative_rms\":" << state_relative_rms
              << ",\"positions_valid\":"
              << (positions_valid ? "true" : "false")
              << ",\"invalid_positions\":" << invalid_positions
              << ",\"minimum_position\":" << minimum_position
              << ",\"maximum_position\":" << maximum_position
              << ",\"windowed_median_us\":" << Median(window_us)
              << ",\"fp32_median_us\":" << Median(fp32_us)
              << ",\"gate\":\"" << (pass ? "pass" : "fail") << "\"}\n";

    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    stream = nullptr;
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    device_set = false;
    Check(aclFinalize(), "aclFinalize");
    initialized = false;
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "GDN State Codec windowed GDN probe failed: " << error.what()
              << '\n';
    if (stream != nullptr) (void)aclrtSynchronizeStream(stream);
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(device);
    if (initialized) (void)aclFinalize();
    return 1;
  }
}

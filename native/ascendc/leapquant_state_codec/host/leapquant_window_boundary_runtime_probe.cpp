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

#include "statecentric/leapquant_windowed_gdn_kernel.h"

namespace {
constexpr std::size_t kStates = 32;
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
  Check(aclrtMemcpy(destination.data(), destination.bytes(), source.data(),
                    source.size() * sizeof(Type), ACL_MEMCPY_HOST_TO_DEVICE),
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

std::vector<float> ReconstructBoundary(
    const std::vector<std::int8_t>& residual,
    const std::vector<float>& smoothing, const std::vector<float>& scales,
    const std::vector<std::uint16_t>& keys,
    const std::vector<std::uint16_t>& values) {
  std::vector<float> dense(kStates * kElements);
  for (std::size_t state = 0; state < kStates; ++state) {
    for (std::size_t row = 0; row < kD; ++row) {
      for (std::size_t column = 0; column < kD; ++column) {
        const std::size_t cell = state * kElements + row * kD + column;
        float value = static_cast<float>(residual[cell]) / 127.0F *
                      smoothing[state * kD + row] *
                      scales[state * kD + column];
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          value += Float(keys[(state * kD + row) * kRank + rank]) *
                   Float(values[(state * kD + column) * kRank + rank]);
        }
        dense[cell] = value;
      }
    }
  }
  return dense;
}

std::vector<float> ApplyRecords(
    std::vector<float> dense, const std::vector<float>& record_decay,
    const std::vector<std::uint16_t>& record_keys,
    const std::vector<std::uint16_t>& record_corrections) {
  for (std::size_t state = 0; state < kStates; ++state) {
    for (std::size_t record = 0; record < kWindow; ++record) {
      const std::size_t base = (state * kWindow + record) * kD;
      for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t column = 0; column < kD; ++column) {
          const std::size_t cell = state * kElements + row * kD + column;
          dense[cell] = record_decay[state * kWindow + record] * dense[cell] +
                        Float(record_keys[base + row]) *
                            Float(record_corrections[base + column]);
        }
      }
    }
  }
  return dense;
}

std::vector<float> QuantizeWithoutCompensators(
    const std::vector<float>& dense) {
  std::vector<float> output(dense.size());
  std::vector<float> smoothing(kStates * kD);
  for (std::size_t state = 0; state < kStates; ++state) {
    for (std::size_t row = 0; row < kD; ++row) {
      double mean_absolute = 0.0;
      for (std::size_t column = 0; column < kD; ++column) {
        mean_absolute +=
            std::abs(dense[state * kElements + row * kD + column]);
      }
      smoothing[state * kD + row] = static_cast<float>(
          std::sqrt(std::max(mean_absolute / static_cast<double>(kD), 1e-8)));
    }
    for (std::size_t column = 0; column < kD; ++column) {
      float maximum = 0.0F;
      for (std::size_t row = 0; row < kD; ++row) {
        maximum = std::max(
            maximum,
            std::abs(dense[state * kElements + row * kD + column] /
                     smoothing[state * kD + row]));
      }
      for (std::size_t row = 0; row < kD; ++row) {
        const std::size_t cell = state * kElements + row * kD + column;
        const float scaled = maximum == 0.0F
                                 ? 0.0F
                                 : dense[cell] /
                                       smoothing[state * kD + row] / maximum *
                                       127.0F;
        const long quantized = std::clamp(std::lround(scaled), -127L, 127L);
        output[cell] = static_cast<float>(quantized) / 127.0F *
                       smoothing[state * kD + row] * maximum;
      }
    }
  }
  return output;
}

double RelativeRms(const std::vector<float>& actual,
                   const std::vector<float>& reference) {
  double error = 0.0;
  double energy = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference = actual[index] - reference[index];
    error += difference * difference;
    energy += static_cast<double>(reference[index]) * reference[index];
  }
  return std::sqrt(error / energy);
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  bool initialized = false;
  bool device_set = false;
  int device = -1;
  aclrtStream stream = nullptr;
  try {
    if (argc < 2 || argc > 3) {
      throw std::invalid_argument("usage: probe DEVICE [POWER_ITERATIONS]");
    }
    device = std::stoi(argv[1]);
    const std::uint32_t power_iterations =
        argc == 3 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 32U;
    if (power_iterations > 32) {
      throw std::invalid_argument("POWER_ITERATIONS must be in [1,32]");
    }
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<std::int8_t> residual(kStates * kElements);
    std::vector<float> smoothing(kStates * kD);
    std::vector<float> scales(kStates * kD);
    std::vector<std::uint16_t> keys(kStates * kD * kRank);
    std::vector<std::uint16_t> values(kStates * kD * kRank);
    std::vector<float> record_decay(kStates * kWindow);
    std::vector<std::uint16_t> record_keys(kStates * kWindow * kD);
    std::vector<std::uint16_t> record_corrections(kStates * kWindow * kD);
    std::vector<std::uint32_t> positions(kStates * kPositionStride, 0);
    for (std::size_t state = 0; state < kStates; ++state) {
      positions[state * kPositionStride] = kWindow;
      for (std::size_t axis = 0; axis < kD; ++axis) {
        smoothing[state * kD + axis] =
            0.75F + 0.002F * static_cast<float>((state + axis) % 41);
        scales[state * kD + axis] =
            0.11F + 0.001F * static_cast<float>((3 * state + axis) % 29);
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          keys[(state * kD + axis) * kRank + rank] = Half(
              0.008F * std::sin(static_cast<float>(state + axis + rank + 1) *
                               0.071F));
          values[(state * kD + axis) * kRank + rank] = Half(
              0.009F * std::cos(static_cast<float>(state + axis + rank + 2) *
                               0.067F));
        }
      }
      for (std::size_t index = 0; index < kElements; ++index) {
        residual[state * kElements + index] = static_cast<std::int8_t>(
            static_cast<int>((index * 17 + state * 11) % 255) - 127);
      }
      for (std::size_t record = 0; record < kWindow; ++record) {
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
    const auto dense_before = ApplyRecords(
        ReconstructBoundary(residual, smoothing, scales, keys, values),
        record_decay, record_keys, record_corrections);
    const auto no_compensator = QuantizeWithoutCompensators(dense_before);

    DeviceBuffer<std::int8_t> residual_d(residual.size());
    DeviceBuffer<float> smoothing_d(smoothing.size());
    DeviceBuffer<float> scales_d(scales.size());
    DeviceBuffer<std::uint16_t> keys_d(keys.size());
    DeviceBuffer<std::uint16_t> values_d(values.size());
    DeviceBuffer<float> record_decay_d(record_decay.size());
    DeviceBuffer<std::uint16_t> record_keys_d(record_keys.size());
    DeviceBuffer<std::uint16_t> record_corrections_d(
        record_corrections.size());
    DeviceBuffer<std::uint32_t> positions_d(positions.size());
    DeviceBuffer<std::int8_t> next_residual_d(residual.size());
    DeviceBuffer<float> next_smoothing_d(smoothing.size());
    DeviceBuffer<float> next_scales_d(scales.size());
    DeviceBuffer<std::uint16_t> next_keys_d(keys.size());
    DeviceBuffer<std::uint16_t> next_values_d(values.size());
    DeviceBuffer<float> residual_workspace_d(residual.size());
    ToDevice(residual_d, residual);
    ToDevice(smoothing_d, smoothing);
    ToDevice(scales_d, scales);
    ToDevice(keys_d, keys);
    ToDevice(values_d, values);
    auto restore = [&] {
      ToDevice(record_decay_d, record_decay);
      ToDevice(record_keys_d, record_keys);
      ToDevice(record_corrections_d, record_corrections);
      ToDevice(positions_d, positions);
    };
    auto launch = [&] {
      if (statecentric_leapquant_window_boundary_launch_v1(
              stream, residual_d.data(), smoothing_d.data(), scales_d.data(),
              keys_d.data(), values_d.data(), record_decay_d.data(),
              record_keys_d.data(), record_corrections_d.data(),
              positions_d.data(), next_residual_d.data(),
              next_smoothing_d.data(), next_scales_d.data(),
              next_keys_d.data(), next_values_d.data(),
              residual_workspace_d.data(), kStates,
              power_iterations) != 0) {
        throw std::runtime_error("boundary launch admission failed");
      }
    };
    restore();
    launch();
    Check(aclrtSynchronizeStream(stream), "sync exactness");
    const auto next_residual = FromDevice(next_residual_d, residual.size());
    const auto next_smoothing = FromDevice(next_smoothing_d, smoothing.size());
    const auto next_scales = FromDevice(next_scales_d, scales.size());
    const auto next_keys = FromDevice(next_keys_d, keys.size());
    const auto next_values = FromDevice(next_values_d, values.size());
    const auto updated_positions = FromDevice(positions_d, positions.size());
    const auto cleared_decay = FromDevice(record_decay_d, record_decay.size());
    const auto cleared_keys = FromDevice(record_keys_d, record_keys.size());
    const auto cleared_corrections =
        FromDevice(record_corrections_d, record_corrections.size());
    const auto dense_after = ReconstructBoundary(
        next_residual, next_smoothing, next_scales, next_keys, next_values);
    const double boundary_error = RelativeRms(dense_after, dense_before);
    const double no_compensator_error =
        RelativeRms(no_compensator, dense_before);
    bool lifecycle_valid = true;
    for (std::size_t state = 0; state < kStates; ++state) {
      lifecycle_valid &= updated_positions[state * kPositionStride] == 0;
    }
    lifecycle_valid &= std::all_of(cleared_decay.begin(), cleared_decay.end(),
                                   [](float value) { return value == 0.0F; });
    lifecycle_valid &= std::all_of(cleared_keys.begin(), cleared_keys.end(),
                                   [](std::uint16_t value) {
                                     return value == Half(0.0F);
                                   });
    lifecycle_valid &= std::all_of(
        cleared_corrections.begin(), cleared_corrections.end(),
        [](std::uint16_t value) { return value == Half(0.0F); });

    std::vector<double> elapsed_us;
    for (int repeat = 0; repeat < 3; ++repeat) {
      restore();
      Check(aclrtSynchronizeStream(stream), "sync before timing");
      const auto begin = std::chrono::steady_clock::now();
      launch();
      Check(aclrtSynchronizeStream(stream), "sync after timing");
      elapsed_us.push_back(std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - begin)
                               .count());
    }
    const bool accuracy_valid = power_iterations == 0
                                    ? boundary_error < 2.0e-2
                                    : boundary_error < no_compensator_error;
    const bool pass = lifecycle_valid && std::isfinite(boundary_error) &&
                      accuracy_valid;
    std::cout << "{\"device\":" << device << ",\"states\":" << kStates
              << ",\"power_iterations\":" << power_iterations
              << ",\"boundary_relative_rms\":" << boundary_error
              << ",\"no_compensator_relative_rms\":"
              << no_compensator_error << ",\"lifecycle_valid\":"
              << (lifecycle_valid ? "true" : "false")
              << ",\"boundary_median_us\":" << Median(elapsed_us)
              << ",\"gate\":\"" << (pass ? "pass" : "fail") << "\"}\n";

    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    stream = nullptr;
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    device_set = false;
    Check(aclFinalize(), "aclFinalize");
    initialized = false;
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "LeapQuant boundary probe failed: " << error.what() << '\n';
    if (stream != nullptr) (void)aclrtSynchronizeStream(stream);
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(device);
    if (initialized) (void)aclFinalize();
    return 1;
  }
}

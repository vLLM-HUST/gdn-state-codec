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

#include "statecentric/gdn_state_codec_kernel.h"

namespace {

constexpr std::size_t kKeys = 128;
constexpr std::size_t kValues = 128;
constexpr std::size_t kRank = 4;

void Check(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(status));
  }
}

template <class Type>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    Check(aclrtMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(Type),
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

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <class Launch>
double Measure(aclrtStream stream, Launch&& launch) {
  Check(aclrtSynchronizeStream(stream), "sync before");
  const auto begin = std::chrono::steady_clock::now();
  launch();
  Check(aclrtSynchronizeStream(stream), "sync after");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  bool initialized = false;
  bool device_set = false;
  int device = -1;
  aclrtStream stream = nullptr;
  try {
    if (argc != 2) throw std::invalid_argument("usage: probe DEVICE");
    device = std::stoi(argv[1]);
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<float> residual(kKeys * kValues);
    std::vector<float> compensator_keys(kKeys * kRank);
    std::vector<float> compensator_values(kValues * kRank);
    for (std::size_t index = 0; index < residual.size(); ++index) {
      residual[index] = 0.2F * std::sin(static_cast<float>(index + 1) * 0.017F);
      if (index % 521 == 0) residual[index] += 2.0F;
    }
    for (std::size_t index = 0; index < compensator_keys.size(); ++index) {
      compensator_keys[index] =
          0.03F * std::sin(static_cast<float>(index + 1) * 0.07F);
    }
    for (std::size_t index = 0; index < compensator_values.size(); ++index) {
      compensator_values[index] =
          0.04F * std::cos(static_cast<float>(index + 1) * 0.09F);
    }

    DeviceBuffer<float> residual_device(residual.size());
    DeviceBuffer<std::int8_t> quantized_device(residual.size());
    DeviceBuffer<float> smoothing_device(kKeys);
    DeviceBuffer<float> scales_device(kValues);
    DeviceBuffer<float> keys_device(compensator_keys.size());
    DeviceBuffer<float> values_device(compensator_values.size());
    DeviceBuffer<float> dense_device(residual.size());
    ToDevice(residual_device, residual);
    ToDevice(keys_device, compensator_keys);
    ToDevice(values_device, compensator_values);

    auto quantize = [&] {
      if (statecentric_gdn_state_codec_smoothing_launch_v1(
              stream, residual_device.data(), smoothing_device.data()) != 0 ||
          statecentric_gdn_state_codec_quantize_launch_v1(
              stream, residual_device.data(), smoothing_device.data(),
              quantized_device.data(), scales_device.data()) != 0) {
        throw std::runtime_error("quantize admission failed");
      }
    };
    auto dequantize = [&] {
      if (statecentric_gdn_state_codec_dequantize_launch_v1(
              stream, quantized_device.data(), smoothing_device.data(),
              scales_device.data(), keys_device.data(), values_device.data(),
              dense_device.data()) != 0) {
        throw std::runtime_error("dequantize admission failed");
      }
    };
    quantize();
    dequantize();
    Check(aclrtSynchronizeStream(stream), "sync exactness");

    const auto quantized = FromDevice(quantized_device, residual.size());
    const auto smoothing = FromDevice(smoothing_device, kKeys);
    const auto scales = FromDevice(scales_device, kValues);
    const auto dense = FromDevice(dense_device, residual.size());
    std::size_t quantized_mismatches = 0;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t row = 0; row < kKeys; ++row) {
      double mean_absolute = 0.0;
      for (std::size_t column = 0; column < kValues; ++column) {
        mean_absolute += std::abs(residual[row * kValues + column]);
      }
      const float expected_smoothing = std::sqrt(std::max(
          static_cast<float>(mean_absolute / kValues), 1.0e-8F));
      if (std::abs(smoothing[row] - expected_smoothing) > 2.0e-5F) {
        throw std::runtime_error(
            "smoothing differs at row " + std::to_string(row) +
            ": device=" + std::to_string(smoothing[row]) +
            ", reference=" + std::to_string(expected_smoothing));
      }
    }
    for (std::size_t column = 0; column < kValues; ++column) {
      float maximum = 0.0F;
      for (std::size_t row = 0; row < kKeys; ++row) {
        maximum = std::max(maximum,
                           std::abs(residual[row * kValues + column] /
                                    smoothing[row]));
      }
      if (std::abs(scales[column] - maximum) > 2.0e-5F) {
        throw std::runtime_error(
            "value scale differs at column " + std::to_string(column) +
            ": device=" + std::to_string(scales[column]) +
            ", reference=" + std::to_string(maximum));
      }
      for (std::size_t row = 0; row < kKeys; ++row) {
        const float scaled = maximum == 0.0F
                                 ? 0.0F
                                 : residual[row * kValues + column] /
                                       smoothing[row] / maximum * 127.0F;
        const auto expected = static_cast<std::int8_t>(
            std::clamp(std::lround(scaled), -127L, 127L));
        quantized_mismatches +=
            expected == quantized[row * kValues + column] ? 0U : 1U;
      }
    }
    for (std::size_t row = 0; row < kKeys; ++row) {
      for (std::size_t column = 0; column < kValues; ++column) {
        float expected = residual[row * kValues + column];
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          expected += compensator_keys[row * kRank + rank] *
                      compensator_values[column * kRank + rank];
        }
        const double difference = dense[row * kValues + column] - expected;
        squared_error += difference * difference;
        squared_reference += static_cast<double>(expected) * expected;
      }
    }

    for (int warmup = 0; warmup < 5; ++warmup) {
      quantize();
      dequantize();
    }
    Check(aclrtSynchronizeStream(stream), "sync warmup");
    std::vector<double> quantize_us;
    std::vector<double> dequantize_us;
    for (int repeat = 0; repeat < 31; ++repeat) {
      if (repeat % 2 == 0) {
        quantize_us.push_back(Measure(stream, quantize));
        dequantize_us.push_back(Measure(stream, dequantize));
      } else {
        dequantize_us.push_back(Measure(stream, dequantize));
        quantize_us.push_back(Measure(stream, quantize));
      }
    }
    std::cout << "{\"device\":" << device
              << ",\"geometry\":\"128x128\",\"rank\":4"
              << ",\"quantized_mismatches\":" << quantized_mismatches
              << ",\"relative_rms\":"
              << std::sqrt(squared_error / squared_reference)
              << ",\"quantize_median_us\":" << Median(quantize_us)
              << ",\"dequantize_median_us\":" << Median(dequantize_us)
              << "}\n";

    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    stream = nullptr;
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    device_set = false;
    Check(aclFinalize(), "aclFinalize");
    initialized = false;
    return quantized_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "GDN State Codec probe failed: " << error.what()
              << '\n';
    if (stream != nullptr) (void)aclrtSynchronizeStream(stream);
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(device);
    if (initialized) (void)aclFinalize();
    return 1;
  }
}

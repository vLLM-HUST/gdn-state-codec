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
constexpr std::size_t kStates = 16;
constexpr std::size_t kKeys = 128;
constexpr std::size_t kValues = 128;
constexpr std::size_t kElements = kStates * kKeys * kValues;

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

    std::vector<float> provider(kElements);
    for (std::size_t index = 0; index < provider.size(); ++index) {
      provider[index] =
          0.4F * std::sin(static_cast<float>(index + 1) * 0.017F) +
          0.2F * std::cos(static_cast<float>(index + 3) * 0.031F);
      if (index % 521 == 0) provider[index] += 2.0F;
    }
    DeviceBuffer<float> provider_device(provider.size());
    DeviceBuffer<float> smoothing_device(kStates * kKeys);
    DeviceBuffer<float> scales_device(kStates * kValues);
    DeviceBuffer<std::int8_t> quantized_device(provider.size());
    ToDevice(provider_device, provider);

    auto launch = [&] {
      if (statecentric_gdn_state_codec_provider_smoothing_launch_v2(
              stream, provider_device.data(), smoothing_device.data(),
              kStates) != 0 ||
          statecentric_gdn_state_codec_provider_quantize_launch_v2(
              stream, provider_device.data(), smoothing_device.data(),
              quantized_device.data(), scales_device.data(), kStates) != 0) {
        throw std::runtime_error("provider quantize admission failed");
      }
    };
    launch();
    Check(aclrtSynchronizeStream(stream), "sync exactness");
    const auto smoothing =
        FromDevice(smoothing_device, kStates * kKeys);
    const auto scales = FromDevice(scales_device, kStates * kValues);
    const auto quantized = FromDevice(quantized_device, provider.size());

    double squared_error = 0.0;
    double squared_reference = 0.0;
    std::size_t quantized_mismatches = 0;
    for (std::size_t state = 0; state < kStates; ++state) {
      std::vector<float> expected_smoothing(kKeys);
      for (std::size_t key = 0; key < kKeys; ++key) {
        double sum = 0.0;
        for (std::size_t value = 0; value < kValues; ++value) {
          const auto index =
              (state * kValues + value) * kKeys + key;
          sum += std::abs(provider[index]);
        }
        expected_smoothing[key] =
            std::sqrt(std::max(static_cast<float>(sum / kValues), 1.0e-8F));
        if (std::abs(smoothing[state * kKeys + key] -
                     expected_smoothing[key]) > 2.0e-5F) {
          throw std::runtime_error("provider smoothing mismatch");
        }
      }
      for (std::size_t value = 0; value < kValues; ++value) {
        float maximum = 0.0F;
        for (std::size_t key = 0; key < kKeys; ++key) {
          const auto index =
              (state * kValues + value) * kKeys + key;
          maximum = std::max(
              maximum, std::abs(provider[index] / expected_smoothing[key]));
        }
        if (std::abs(scales[state * kValues + value] - maximum) > 2.0e-5F) {
          throw std::runtime_error("provider value-scale mismatch");
        }
        for (std::size_t key = 0; key < kKeys; ++key) {
          const auto index =
              (state * kValues + value) * kKeys + key;
          const float scaled = maximum == 0.0F
              ? 0.0F
              : provider[index] / expected_smoothing[key] / maximum * 127.0F;
          const auto expected = static_cast<std::int8_t>(
              std::clamp(std::lround(scaled), -127L, 127L));
          quantized_mismatches += expected == quantized[index] ? 0U : 1U;
          const double reconstructed =
              static_cast<float>(quantized[index]) / 127.0F *
              smoothing[state * kKeys + key] *
              scales[state * kValues + value];
          const double difference = reconstructed - provider[index];
          squared_error += difference * difference;
          squared_reference +=
              static_cast<double>(provider[index]) * provider[index];
        }
      }
    }
    const double relative_rms =
        std::sqrt(squared_error / squared_reference);
    if (relative_rms >= 0.02) {
      throw std::runtime_error(
          "provider reconstruction gate failed: " +
          std::to_string(relative_rms));
    }

    for (int warmup = 0; warmup < 5; ++warmup) launch();
    Check(aclrtSynchronizeStream(stream), "sync warmup");
    std::vector<double> elapsed_us;
    for (int repeat = 0; repeat < 101; ++repeat) {
      Check(aclrtSynchronizeStream(stream), "sync before");
      const auto begin = std::chrono::steady_clock::now();
      launch();
      Check(aclrtSynchronizeStream(stream), "sync after");
      elapsed_us.push_back(std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - begin)
                               .count());
    }
    std::cout << "{\"device\":" << device << ",\"states\":" << kStates
              << ",\"geometry\":\"128x128\""
              << ",\"quantized_mismatches\":" << quantized_mismatches
              << ",\"relative_rms\":" << relative_rms
              << ",\"median_us\":" << Median(elapsed_us) << "}\n";

    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    stream = nullptr;
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    device_set = false;
    Check(aclFinalize(), "aclFinalize");
    initialized = false;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GDN provider quantize probe failed: " << error.what()
              << '\n';
    if (stream != nullptr) (void)aclrtSynchronizeStream(stream);
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(device);
    if (initialized) (void)aclFinalize();
    return 1;
  }
}

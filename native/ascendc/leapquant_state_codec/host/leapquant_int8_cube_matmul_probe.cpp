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
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr std::size_t kStates = 32;
constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 16;

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
void ToDevice(DeviceBuffer<Type>& destination, const std::vector<Type>& source) {
  Check(aclrtMemcpy(destination.data(), destination.bytes(), source.data(),
                    source.size() * sizeof(Type), ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy H2D");
}

template <class Type>
std::vector<Type> FromDevice(const DeviceBuffer<Type>& source) {
  std::vector<Type> output(source.bytes() / sizeof(Type));
  Check(aclrtMemcpy(output.data(), output.size() * sizeof(Type), source.data(),
                    source.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy D2H");
  return output;
}

std::vector<std::uint8_t> MakeTiling(
    const platform_ascendc::PlatformAscendC& platform, bool transpose_a) {
  optiling::TCubeTiling tiling;
  matmul_tiling::MultiCoreMatmulTiling generator(platform);
  generator.SetDim(static_cast<int32_t>(platform.GetCoreNumAiv()));
  generator.SetAType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT8, transpose_a);
  generator.SetBType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT8, false);
  generator.SetCType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT32);
  generator.SetOrgShape(kRows, kColumns, kRows);
  generator.SetShape(kRows, kColumns, kRows);
  generator.SetBias(false);
  generator.SetBufferSpace(-1, -1, -1);
  if (generator.GetTiling(tiling) < 0) {
    throw std::runtime_error("INT8 matmul tiling failed");
  }
  std::vector<std::uint8_t> output(tiling.GetDataSize());
  tiling.SaveToBuffer(output.data(), output.size());
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
  aclrtStream stream = nullptr;
  int device = -1;
  try {
    if (argc != 2) throw std::invalid_argument("usage: probe DEVICE");
    device = std::stoi(argv[1]);
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    auto* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B2");
    if (platform == nullptr) throw std::runtime_error("platform unavailable");

    std::vector<std::int8_t> matrix(kStates * kRows * kRows);
    std::vector<std::int8_t> vector(kStates * kRows * kColumns, 0);
    for (std::size_t state = 0; state < kStates; ++state) {
      for (std::size_t row = 0; row < kRows; ++row) {
        vector[(state * kRows + row) * kColumns] =
            static_cast<std::int8_t>(
                static_cast<int>((state + 3) * (row + 5) % 29) - 14);
        for (std::size_t column = 0; column < kRows; ++column) {
          matrix[(state * kRows + row) * kRows + column] =
              static_cast<std::int8_t>(static_cast<int>(
                  (state * 31 + row * 7 + column * 11 + 1) % 127) - 63);
        }
      }
    }
    DeviceBuffer<std::int8_t> matrix_device(matrix.size());
    DeviceBuffer<std::int8_t> vector_device(vector.size());
    DeviceBuffer<std::int32_t> output_device(vector.size());
    const auto forward_tiling = MakeTiling(*platform, false);
    const auto transpose_tiling = MakeTiling(*platform, true);
    DeviceBuffer<std::uint8_t> forward_tiling_device(forward_tiling.size());
    DeviceBuffer<std::uint8_t> transpose_tiling_device(transpose_tiling.size());
    DeviceBuffer<std::uint8_t> workspace_device(
        platform->GetLibApiWorkSpaceSize());
    ToDevice(matrix_device, matrix);
    ToDevice(vector_device, vector);
    ToDevice(forward_tiling_device, forward_tiling);
    ToDevice(transpose_tiling_device, transpose_tiling);
    constexpr std::uint32_t kBlocks = (kStates + 1U) / 2U;

    auto run = [&](bool transpose) {
      if (statecentric_leapquant_int8_cube_matmul_launch_v1(
              stream, matrix_device.data(), vector_device.data(),
              output_device.data(), kStates, transpose ? 1U : 0U,
              workspace_device.data(),
              transpose ? transpose_tiling_device.data()
                        : forward_tiling_device.data(),
              kBlocks) != 0) {
        throw std::runtime_error("INT8 Cube launch admission failed");
      }
      Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    };
    run(false);
    std::vector<double> timings;
    for (int repetition = 0; repetition < 11; ++repetition) {
      const auto start = std::chrono::steady_clock::now();
      run(false);
      const auto stop = std::chrono::steady_clock::now();
      timings.push_back(std::chrono::duration<double, std::micro>(stop - start)
                            .count());
    }
    const auto actual_forward = FromDevice(output_device);
    run(true);
    const auto actual_transpose = FromDevice(output_device);
    std::int64_t maximum_error = 0;
    std::int64_t padding_maximum = 0;
    for (std::size_t state = 0; state < kStates; ++state) {
      for (std::size_t row = 0; row < kRows; ++row) {
        std::int32_t forward = 0;
        std::int32_t transpose = 0;
        for (std::size_t inner = 0; inner < kRows; ++inner) {
          const auto input = static_cast<std::int32_t>(
              vector[(state * kRows + inner) * kColumns]);
          forward += static_cast<std::int32_t>(
                         matrix[(state * kRows + row) * kRows + inner]) *
                     input;
          transpose += static_cast<std::int32_t>(
                           matrix[(state * kRows + inner) * kRows + row]) *
                       input;
        }
        const std::size_t base = (state * kRows + row) * kColumns;
        maximum_error = std::max<std::int64_t>(
            maximum_error,
            std::llabs(static_cast<long long>(actual_forward[base]) - forward));
        maximum_error = std::max<std::int64_t>(
            maximum_error,
            std::llabs(static_cast<long long>(actual_transpose[base]) -
                       transpose));
        for (std::size_t column = 1; column < kColumns; ++column) {
          padding_maximum = std::max<std::int64_t>(
              padding_maximum,
              std::llabs(static_cast<long long>(
                  actual_transpose[base + column])));
        }
      }
    }
    const bool passed = maximum_error == 0 && padding_maximum == 0;
    std::cout << "evidence_label=synthetic-int8-cube-abi-gate\n"
              << "states=" << kStates << '\n'
              << "maximum_integer_error=" << maximum_error << '\n'
              << "padding_max_abs=" << padding_maximum << '\n'
              << "median_forward_us=" << Median(timings) << '\n'
              << "gate=" << (passed ? "pass" : "fail") << '\n';

    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    stream = nullptr;
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    device_set = false;
    Check(aclFinalize(), "aclFinalize");
    initialized = false;
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(device);
    if (initialized) (void)aclFinalize();
    return 1;
  }
}

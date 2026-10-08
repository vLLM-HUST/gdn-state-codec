#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "statecentric/leapquant_windowed_gdn_kernel.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr std::size_t kM = 128;
constexpr std::size_t kN = 16;
constexpr std::size_t kK = 128;

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
    if (argc < 2 || argc > 4) {
      throw std::invalid_argument("usage: probe DEVICE [STATES] [TRANSPOSE_A]");
    }
    device = std::stoi(argv[1]);
    const std::size_t states =
        argc >= 3 ? static_cast<std::size_t>(std::stoul(argv[2])) : 1U;
    if (states == 0 || states > 128) {
      throw std::invalid_argument("STATES must be in [1,128]");
    }
    const bool transpose_a = argc == 4 ? std::stoul(argv[3]) != 0U : false;
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    auto* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B2");
    if (platform == nullptr) throw std::runtime_error("platform unavailable");
    optiling::TCubeTiling tiling;
    matmul_tiling::MultiCoreMatmulTiling generator(*platform);
    generator.SetDim(static_cast<int32_t>(platform->GetCoreNumAiv()));
    generator.SetAType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16, transpose_a);
    generator.SetBType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16, false);
    generator.SetCType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT);
    generator.SetOrgShape(kM, kN, kK);
    generator.SetShape(kM, kN, kK);
    generator.SetBias(false);
    generator.SetBufferSpace(-1, -1, -1);
    if (generator.GetTiling(tiling) < 0) {
      throw std::runtime_error("matmul tiling failed");
    }

    std::vector<std::uint16_t> a(states * kM * kK);
    std::vector<std::uint16_t> b(states * kK * kN);
    std::vector<float> reference(states * kM * kN, 0.0F);
    for (std::size_t state = 0; state < states; ++state) {
      for (std::size_t row = 0; row < kM; ++row) {
        for (std::size_t inner = 0; inner < kK; ++inner) {
          a[(state * kM + row) * kK + inner] = Half(
              0.07F * std::sin(
                          static_cast<float>(state * 29 + row * 13 + inner + 1) *
                          0.017F));
        }
      }
      for (std::size_t inner = 0; inner < kK; ++inner) {
        for (std::size_t column = 0; column < kN; ++column) {
          b[(state * kK + inner) * kN + column] = Half(
              0.09F * std::cos(
                          static_cast<float>(state * 31 + inner * 7 + column + 2) *
                          0.019F));
        }
      }
      for (std::size_t row = 0; row < kM; ++row) {
        for (std::size_t column = 0; column < kN; ++column) {
          double value = 0.0;
          for (std::size_t inner = 0; inner < kK; ++inner) {
            const std::size_t a_row = transpose_a ? inner : row;
            const std::size_t a_column = transpose_a ? row : inner;
            value += static_cast<double>(
                         Float(a[(state * kM + a_row) * kK + a_column])) *
                     Float(b[(state * kK + inner) * kN + column]);
          }
          reference[(state * kM + row) * kN + column] =
              static_cast<float>(value);
        }
      }
    }

    const std::size_t tiling_bytes = tiling.GetDataSize();
    std::vector<std::uint8_t> tiling_host(tiling_bytes);
    tiling.SaveToBuffer(tiling_host.data(), tiling_bytes);
    DeviceBuffer<std::uint16_t> a_device(a.size());
    DeviceBuffer<std::uint16_t> b_device(b.size());
    DeviceBuffer<float> c_device(reference.size());
    DeviceBuffer<std::uint8_t> tiling_device(tiling_bytes);
    const std::size_t workspace_bytes = platform->GetLibApiWorkSpaceSize();
    DeviceBuffer<std::uint8_t> workspace_device(workspace_bytes);
    ToDevice(a_device, a);
    ToDevice(b_device, b);
    ToDevice(tiling_device, tiling_host);

    const auto used_core_num =
        static_cast<uint32_t>(tiling.get_usedCoreNum());
    const uint32_t blocks = static_cast<uint32_t>((states + 1U) / 2U);
    TCubeTiling device_tiling{};
    if (tiling_bytes < sizeof(device_tiling)) {
      throw std::runtime_error("serialized tiling is smaller than TCubeTiling");
    }
    std::memcpy(&device_tiling, tiling_host.data(), sizeof(device_tiling));
    std::cout << "prelaunch_used_core_num=" << device_tiling.usedCoreNum
              << " M=" << device_tiling.M << " N=" << device_tiling.N
              << " Ka=" << device_tiling.Ka << " Kb=" << device_tiling.Kb
              << " singleCoreM=" << device_tiling.singleCoreM
              << " singleCoreN=" << device_tiling.singleCoreN
              << " singleCoreK=" << device_tiling.singleCoreK
              << " baseM=" << device_tiling.baseM
              << " baseN=" << device_tiling.baseN
              << " baseK=" << device_tiling.baseK
              << " depthA1=" << device_tiling.depthA1
              << " depthB1=" << device_tiling.depthB1
              << " stepM=" << device_tiling.stepM
              << " stepN=" << device_tiling.stepN
              << " workspace_bytes=" << workspace_bytes << std::endl;
    auto launch = [&] {
      if (statecentric_leapquant_cube_matmul_probe_launch_v1(
              stream, a_device.data(), b_device.data(), c_device.data(),
              static_cast<uint32_t>(states), transpose_a ? 1U : 0U,
              workspace_device.data(), tiling_device.data(), blocks) != 0) {
        throw std::runtime_error("cube launch admission failed");
      }
    };
    launch();
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    std::vector<double> timings;
    for (int iteration = 0; iteration < 21; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      launch();
      Check(aclrtSynchronizeStream(stream), "timed synchronize");
      const auto stop = std::chrono::steady_clock::now();
      timings.push_back(std::chrono::duration<double, std::micro>(stop - start)
                            .count());
    }
    std::vector<float> actual(reference.size());
    Check(aclrtMemcpy(actual.data(), actual.size() * sizeof(float),
                      c_device.data(), c_device.bytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy D2H");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const double error = actual[index] - reference[index];
      squared_error += error * error;
      squared_reference +=
          static_cast<double>(reference[index]) * reference[index];
      maximum_error = std::max(maximum_error, std::abs(error));
    }
    const double relative_rms = std::sqrt(squared_error / squared_reference);
    const bool passed = relative_rms < 2.0e-3 && maximum_error < 2.0e-3;
    std::cout << "evidence_label=synthetic-cube-abi-probe\n"
              << "shape=" << kM << "x" << kK << "x" << kN << '\n'
              << "states=" << states << '\n'
              << "transpose_a=" << (transpose_a ? 1 : 0) << '\n'
              << "used_core_num=" << used_core_num << '\n'
              << "launch_blocks=" << blocks << '\n'
              << "workspace_bytes=" << workspace_bytes << '\n'
              << "relative_rms=" << relative_rms << '\n'
              << "max_abs_error=" << maximum_error << '\n'
              << "median_us=" << Median(timings) << '\n'
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

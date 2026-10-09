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
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr std::size_t kStates = 32;
constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 16;
constexpr std::size_t kRanks = 4;

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
                     matmul_tiling::DataType::DT_FLOAT16, transpose_a);
  generator.SetBType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16, false);
  generator.SetCType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
  generator.SetOrgShape(kRows, kColumns, kRows);
  generator.SetShape(kRows, kColumns, kRows);
  generator.SetBias(false);
  generator.SetBufferSpace(-1, -1, -1);
  if (generator.GetTiling(tiling) < 0) {
    throw std::runtime_error("matmul tiling failed");
  }
  std::vector<std::uint8_t> output(tiling.GetDataSize());
  tiling.SaveToBuffer(output.data(), output.size());
  return output;
}

void NormalizeColumn(std::vector<std::uint16_t>& vector) {
  for (std::size_t state = 0; state < kStates; ++state) {
    double squared_norm = 0.0;
    for (std::size_t row = 0; row < kRows; ++row) {
      const float value = Float(vector[(state * kRows + row) * kColumns]);
      squared_norm += static_cast<double>(value) * value;
    }
    const float inverse =
        squared_norm > 1.0e-20 ? 1.0F / std::sqrt(squared_norm) : 0.0F;
    for (std::size_t row = 0; row < kRows; ++row) {
      const std::size_t base = (state * kRows + row) * kColumns;
      vector[base] = Half(Float(vector[base]) * inverse);
      for (std::size_t column = 1; column < kColumns; ++column) {
        vector[base + column] = Half(0.0F);
      }
    }
  }
}

std::vector<float> MatmulColumn(const std::vector<std::uint16_t>& matrix,
                                const std::vector<std::uint16_t>& vector,
                                bool transpose) {
  std::vector<float> output(kStates * kRows * kColumns, 0.0F);
  for (std::size_t state = 0; state < kStates; ++state) {
    for (std::size_t row = 0; row < kRows; ++row) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < kRows; ++inner) {
        const std::size_t matrix_row = transpose ? inner : row;
        const std::size_t matrix_column = transpose ? row : inner;
        value += static_cast<double>(Float(matrix[
                     (state * kRows + matrix_row) * kRows + matrix_column])) *
                 Float(vector[(state * kRows + inner) * kColumns]);
      }
      output[(state * kRows + row) * kColumns] = static_cast<float>(value);
    }
  }
  return output;
}

std::vector<std::uint16_t> NormalizeOutput(const std::vector<float>& input) {
  std::vector<std::uint16_t> output(input.size(), Half(0.0F));
  for (std::size_t state = 0; state < kStates; ++state) {
    double squared_norm = 0.0;
    for (std::size_t row = 0; row < kRows; ++row) {
      const float value = input[(state * kRows + row) * kColumns];
      squared_norm += static_cast<double>(value) * value;
    }
    const float inverse =
        squared_norm > 1.0e-20 ? 1.0F / std::sqrt(squared_norm) : 0.0F;
    for (std::size_t row = 0; row < kRows; ++row) {
      const std::size_t index = (state * kRows + row) * kColumns;
      output[index] = Half(input[index] * inverse);
    }
  }
  return output;
}

void ApplyDeflation(std::vector<float>& projected,
                    const std::vector<std::uint16_t>& vector,
                    const std::vector<std::uint16_t>& left_basis,
                    const std::vector<std::uint16_t>& right_basis,
                    std::size_t completed_ranks, bool transpose) {
  for (std::size_t state = 0; state < kStates; ++state) {
    std::vector<double> coefficients(completed_ranks, 0.0);
    for (std::size_t rank = 0; rank < completed_ranks; ++rank) {
      for (std::size_t inner = 0; inner < kRows; ++inner) {
        const std::size_t basis =
            (state * kRanks + rank) * kRows + inner;
        const std::size_t input =
            (state * kRows + inner) * kColumns;
        coefficients[rank] +=
            Float((transpose ? left_basis : right_basis)[basis]) *
            Float(vector[input]);
      }
    }
    for (std::size_t row = 0; row < kRows; ++row) {
      const std::size_t output = (state * kRows + row) * kColumns;
      for (std::size_t rank = 0; rank < completed_ranks; ++rank) {
        const std::size_t basis =
            (state * kRanks + rank) * kRows + row;
        projected[output] -=
            Float((transpose ? right_basis : left_basis)[basis]) *
            static_cast<float>(coefficients[rank]);
      }
    }
  }
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
    if (argc < 2 || argc > 3) {
      throw std::invalid_argument("usage: probe DEVICE [ITERATIONS]");
    }
    device = std::stoi(argv[1]);
    const std::uint32_t iterations =
        argc == 3 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 8U;
    if (iterations == 0 || iterations > 32) {
      throw std::invalid_argument("ITERATIONS must be in [1,32]");
    }
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    auto* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B2");
    if (platform == nullptr) throw std::runtime_error("platform unavailable");
    const auto forward_tiling = MakeTiling(*platform, false);
    const auto transpose_tiling = MakeTiling(*platform, true);
    std::vector<std::uint16_t> matrix(kStates * kRows * kRows);
    const std::size_t vector_elements = kStates * kRows * kColumns;
    std::vector<std::uint16_t> seeds(kRanks * vector_elements, Half(0.0F));
    for (std::size_t state = 0; state < kStates; ++state) {
      for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t rank = 0; rank < kRanks; ++rank) {
          seeds[rank * vector_elements +
                (state * kRows + row) * kColumns] = Half(static_cast<float>(
              static_cast<int>((state + rank + 1) * (row + 3 + rank * 5) %
                               23) -
              11));
        }
        for (std::size_t column = 0; column < kRows; ++column) {
          matrix[(state * kRows + row) * kRows + column] = Half(
              0.045F * std::sin(
                           static_cast<float>(state * 37 + row * 11 + column + 1) *
                           0.013F) +
              (row == column ? 0.08F : 0.0F));
        }
      }
    }
    for (std::size_t rank = 0; rank < kRanks; ++rank) {
      std::vector<std::uint16_t> seed(
          seeds.begin() + static_cast<std::ptrdiff_t>(rank * vector_elements),
          seeds.begin() +
              static_cast<std::ptrdiff_t>((rank + 1) * vector_elements));
      NormalizeColumn(seed);
      std::copy(seed.begin(), seed.end(),
                seeds.begin() +
                    static_cast<std::ptrdiff_t>(rank * vector_elements));
    }
    std::vector<std::uint16_t> reference_left_basis(
        kStates * kRanks * kRows, Half(0.0F));
    std::vector<std::uint16_t> reference_right_basis(
        kStates * kRanks * kRows, Half(0.0F));
    std::vector<float> reference_last_projection;
    for (std::size_t rank = 0; rank < kRanks; ++rank) {
      std::vector<std::uint16_t> reference_right(
          seeds.begin() + static_cast<std::ptrdiff_t>(rank * vector_elements),
          seeds.begin() +
              static_cast<std::ptrdiff_t>((rank + 1) * vector_elements));
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        auto reference_left =
            MatmulColumn(matrix, reference_right, false);
        ApplyDeflation(reference_left, reference_right, reference_left_basis,
                       reference_right_basis, rank, false);
        const auto normalized_left = NormalizeOutput(reference_left);
        auto transposed = MatmulColumn(matrix, normalized_left, true);
        ApplyDeflation(transposed, normalized_left, reference_left_basis,
                       reference_right_basis, rank, true);
        reference_right = NormalizeOutput(transposed);
      }
      reference_last_projection =
          MatmulColumn(matrix, reference_right, false);
      ApplyDeflation(reference_last_projection, reference_right,
                     reference_left_basis, reference_right_basis, rank, false);
      for (std::size_t state = 0; state < kStates; ++state) {
        for (std::size_t row = 0; row < kRows; ++row) {
          const std::size_t vector_index =
              (state * kRows + row) * kColumns;
          const std::size_t basis_index =
              (state * kRanks + rank) * kRows + row;
          reference_left_basis[basis_index] =
              Half(reference_last_projection[vector_index]);
          reference_right_basis[basis_index] = reference_right[vector_index];
        }
      }
    }

    DeviceBuffer<std::uint16_t> matrix_device(matrix.size());
    DeviceBuffer<std::uint16_t> seeds_device(seeds.size());
    DeviceBuffer<std::uint16_t> right_device(vector_elements);
    DeviceBuffer<std::uint16_t> normalized_device(vector_elements);
    DeviceBuffer<float> projected_device(vector_elements);
    DeviceBuffer<std::uint16_t> left_basis_device(
        reference_left_basis.size());
    DeviceBuffer<std::uint16_t> right_basis_device(
        reference_right_basis.size());
    DeviceBuffer<std::uint8_t> forward_tiling_device(forward_tiling.size());
    DeviceBuffer<std::uint8_t> transpose_tiling_device(
        transpose_tiling.size());
    DeviceBuffer<std::uint8_t> workspace_device(
        platform->GetLibApiWorkSpaceSize());
    ToDevice(matrix_device, matrix);
    ToDevice(seeds_device, seeds);
    ToDevice(forward_tiling_device, forward_tiling);
    ToDevice(transpose_tiling_device, transpose_tiling);
    constexpr std::uint32_t kBlocks = (kStates + 1U) / 2U;

    auto cube = [&](const DeviceBuffer<std::uint16_t>& vector,
                    bool transpose) {
      if (statecentric_gdn_state_codec_cube_matmul_probe_launch_v1(
              stream, matrix_device.data(), vector.data(),
              projected_device.data(), kStates, transpose ? 1U : 0U,
              workspace_device.data(),
              transpose ? transpose_tiling_device.data()
                        : forward_tiling_device.data(),
              kBlocks) != 0) {
        throw std::runtime_error("cube launch admission failed");
      }
    };
    auto deflate_normalize = [&](const DeviceBuffer<std::uint16_t>& vector,
                                 DeviceBuffer<std::uint16_t>& output,
                                 std::uint32_t rank, bool transpose) {
      if (statecentric_gdn_state_codec_cube_deflate_normalize_launch_v1(
              stream, projected_device.data(), vector.data(),
              left_basis_device.data(), right_basis_device.data(),
              output.data(), kStates, rank, transpose ? 1U : 0U) != 0) {
        throw std::runtime_error(
            "deflate-normalize launch admission failed");
      }
    };
    auto chain = [&] {
      for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
        Check(aclrtMemcpyAsync(
                  right_device.data(), right_device.bytes(),
                  seeds_device.data() + rank * vector_elements,
                  right_device.bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "seed D2D");
        for (std::uint32_t iteration = 0; iteration < iterations;
             ++iteration) {
          cube(right_device, false);
          deflate_normalize(right_device, normalized_device, rank, false);
          cube(normalized_device, true);
          deflate_normalize(normalized_device, right_device, rank, true);
        }
        cube(right_device, false);
        if (statecentric_gdn_state_codec_cube_commit_rank_launch_v1(
                stream, projected_device.data(), right_device.data(),
                left_basis_device.data(), right_basis_device.data(), kStates,
                rank) != 0) {
          throw std::runtime_error("commit-rank launch admission failed");
        }
      }
    };

    Check(aclrtMemsetAsync(left_basis_device.data(), left_basis_device.bytes(),
                           0, left_basis_device.bytes(), stream),
          "warmup left basis memset");
    Check(aclrtMemsetAsync(right_basis_device.data(),
                           right_basis_device.bytes(), 0,
                           right_basis_device.bytes(), stream),
          "warmup right basis memset");
    chain();
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    std::vector<double> timings;
    for (int repetition = 0; repetition < 11; ++repetition) {
      Check(aclrtMemsetAsync(left_basis_device.data(),
                             left_basis_device.bytes(), 0,
                             left_basis_device.bytes(), stream),
            "timed left basis memset");
      Check(aclrtMemsetAsync(right_basis_device.data(),
                             right_basis_device.bytes(), 0,
                             right_basis_device.bytes(), stream),
            "timed right basis memset");
      const auto start = std::chrono::steady_clock::now();
      chain();
      Check(aclrtSynchronizeStream(stream), "timed synchronize");
      const auto stop = std::chrono::steady_clock::now();
      timings.push_back(std::chrono::duration<double, std::micro>(stop - start)
                            .count());
    }
    const auto actual_left_basis = FromDevice(left_basis_device);
    const auto actual_right_basis = FromDevice(right_basis_device);
    const auto actual_last_projection = FromDevice(projected_device);
    double right_error = 0.0;
    double right_energy = 0.0;
    double left_error = 0.0;
    double left_energy = 0.0;
    double projection_error = 0.0;
    double projection_energy = 0.0;
    double padding_max = 0.0;
    for (std::size_t state = 0; state < kStates; ++state) {
      for (std::size_t rank = 0; rank < kRanks; ++rank) {
        for (std::size_t row = 0; row < kRows; ++row) {
          const std::size_t basis =
              (state * kRanks + rank) * kRows + row;
          const double right_difference = Float(actual_right_basis[basis]) -
                                          Float(reference_right_basis[basis]);
          right_error += right_difference * right_difference;
          right_energy +=
              static_cast<double>(Float(reference_right_basis[basis])) *
              Float(reference_right_basis[basis]);
          const double left_difference = Float(actual_left_basis[basis]) -
                                         Float(reference_left_basis[basis]);
          left_error += left_difference * left_difference;
          left_energy +=
              static_cast<double>(Float(reference_left_basis[basis])) *
              Float(reference_left_basis[basis]);
        }
      }
      for (std::size_t row = 0; row < kRows; ++row) {
        const std::size_t base = (state * kRows + row) * kColumns;
        const double right_difference =
            actual_last_projection[base] - reference_last_projection[base];
        projection_error += right_difference * right_difference;
        projection_energy +=
            static_cast<double>(reference_last_projection[base]) *
            reference_last_projection[base];
        for (std::size_t column = 1; column < kColumns; ++column) {
          padding_max = std::max(
              padding_max,
              std::abs(static_cast<double>(
                  actual_last_projection[base + column])));
        }
      }
    }
    const double right_rms = std::sqrt(right_error / right_energy);
    const double left_rms = std::sqrt(left_error / left_energy);
    const double projection_rms =
        std::sqrt(projection_error / projection_energy);
    const bool passed = right_rms < 3.0e-3 && left_rms < 3.0e-3 &&
                        projection_rms < 3.0e-3 && padding_max < 1.0e-6;
    std::cout << "evidence_label=synthetic-rank4-deflated-power-iteration-gate\n"
              << "states=" << kStates << '\n'
              << "ranks=" << kRanks << '\n'
              << "iterations=" << iterations << '\n'
              << "right_relative_rms=" << right_rms << '\n'
              << "left_relative_rms=" << left_rms << '\n'
              << "last_projection_relative_rms=" << projection_rms << '\n'
              << "padding_max_abs=" << padding_max << '\n'
              << "median_chain_us=" << Median(timings) << '\n'
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

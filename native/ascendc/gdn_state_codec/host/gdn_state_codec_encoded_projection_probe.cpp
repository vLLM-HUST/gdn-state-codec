#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "statecentric/gdn_state_codec_windowed_gdn_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr std::size_t kStates = 32;
constexpr std::size_t kD = 128;
constexpr std::size_t kColumns = 16;
constexpr std::size_t kRank = 4;
constexpr std::size_t kWindow = 16;

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

template <class Type>
std::vector<Type> ReadBinary(const std::filesystem::path& path,
                             std::size_t elements) {
  std::error_code error;
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || bytes != elements * sizeof(Type)) {
    throw std::runtime_error("fixture size mismatch: " + path.string());
  }
  std::vector<Type> output(elements);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(output.data()),
              static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("fixture read failed: " + path.string());
  return output;
}

std::vector<std::uint8_t> MakeTiling(
    const platform_ascendc::PlatformAscendC& platform, bool transpose) {
  optiling::TCubeTiling tiling;
  matmul_tiling::MultiCoreMatmulTiling generator(platform);
  generator.SetDim(static_cast<int32_t>(platform.GetCoreNumAiv()));
  generator.SetAType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT8, transpose);
  generator.SetBType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT8, false);
  generator.SetCType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT32);
  generator.SetOrgShape(kD, kColumns, kD);
  generator.SetShape(kD, kColumns, kD);
  generator.SetBias(false);
  generator.SetBufferSpace(-1, -1, -1);
  if (generator.GetTiling(tiling) < 0) throw std::runtime_error("tiling failed");
  std::vector<std::uint8_t> output(tiling.GetDataSize());
  tiling.SaveToBuffer(output.data(), output.size());
  return output;
}

struct ProjectionResult {
  std::vector<std::uint16_t> normalized;
  std::vector<float> raw;
};

ProjectionResult Project(
    const std::vector<std::int8_t>& residual,
    const std::vector<float>& smoothing, const std::vector<float>& scales,
    const std::vector<std::uint16_t>& old_left,
    const std::vector<std::uint16_t>& old_right,
    const std::vector<float>& record_decay,
    const std::vector<std::uint16_t>& record_left,
    const std::vector<std::uint16_t>& record_right,
    const std::vector<std::uint16_t>& new_left,
    const std::vector<std::uint16_t>& new_right,
    const std::vector<std::uint16_t>& input, std::size_t fitted_ranks,
    bool transpose, bool quantized_input) {
  ProjectionResult result{
      std::vector<std::uint16_t>(kStates * kD * kColumns, Half(0.0F)),
      std::vector<float>(kStates * kD * kColumns, 0.0F)};
  for (std::size_t state = 0; state < kStates; ++state) {
    const std::size_t vector_base = state * kD * kColumns;
    const std::size_t axis_base = state * kD;
    std::vector<float> scaled(kD);
    float maximum = 0.0F;
    for (std::size_t axis = 0; axis < kD; ++axis) {
      scaled[axis] = Float(input[vector_base + axis * kColumns]) *
                     (transpose ? smoothing[axis_base + axis]
                                : scales[axis_base + axis]);
      maximum = std::max(maximum, std::abs(scaled[axis]));
    }
    const float dynamic_scale = maximum > 1.0e-20F ? maximum / 127.0F : 1.0F;
    std::vector<std::int8_t> quantized(kD);
    for (std::size_t axis = 0; axis < kD; ++axis) {
      quantized[axis] = static_cast<std::int8_t>(
          std::clamp(std::round(scaled[axis] / dynamic_scale), -127.0F,
                     127.0F));
    }
    float suffix = 1.0F;
    std::vector<float> record_suffix(kWindow);
    for (std::size_t reverse = kWindow; reverse > 0; --reverse) {
      const std::size_t record = reverse - 1;
      record_suffix[record] = suffix;
      suffix *= record_decay[state * kWindow + record];
    }
    std::vector<float> values(kD, 0.0F);
    std::vector<double> old_coefficients(kRank, 0.0);
    std::vector<double> record_coefficients(kWindow, 0.0);
    std::vector<double> new_coefficients(fitted_ranks, 0.0);
    for (std::size_t inner = 0; inner < kD; ++inner) {
      const float input_value =
          Float(input[vector_base + inner * kColumns]);
      for (std::size_t rank = 0; rank < kRank; ++rank) {
        const std::size_t index = (state * kD + inner) * kRank + rank;
        old_coefficients[rank] +=
            Float((transpose ? old_left : old_right)[index]) * input_value;
      }
      for (std::size_t record = 0; record < kWindow; ++record) {
        const std::size_t index =
            (state * kWindow + record) * kD + inner;
        record_coefficients[record] +=
            Float((transpose ? record_left : record_right)[index]) *
            input_value;
      }
      for (std::size_t rank = 0; rank < fitted_ranks; ++rank) {
        const std::size_t index = (state * kRank + rank) * kD + inner;
        new_coefficients[rank] +=
            Float((transpose ? new_left : new_right)[index]) * input_value;
      }
    }
    for (std::size_t row = 0; row < kD; ++row) {
      double residual_value = 0.0;
      for (std::size_t inner = 0; inner < kD; ++inner) {
        const std::size_t matrix_index =
            (state * kD + (transpose ? inner : row)) * kD +
            (transpose ? row : inner);
        residual_value += static_cast<float>(residual[matrix_index]) *
                          (quantized_input
                               ? static_cast<float>(quantized[inner]) *
                                     dynamic_scale
                               : scaled[inner]);
      }
      values[row] = static_cast<float>(residual_value) / 127.0F *
                    (transpose ? scales[axis_base + row]
                               : smoothing[axis_base + row]) *
                    suffix;
      for (std::size_t rank = 0; rank < kRank; ++rank) {
        const std::size_t output_index =
            (state * kD + row) * kRank + rank;
        values[row] += Float((transpose ? old_right : old_left)[output_index]) *
                       static_cast<float>(old_coefficients[rank]) * suffix;
      }
      for (std::size_t record = 0; record < kWindow; ++record) {
        const std::size_t output_index =
            (state * kWindow + record) * kD + row;
        values[row] +=
            Float((transpose ? record_right : record_left)[output_index]) *
            static_cast<float>(record_coefficients[record]) *
            record_suffix[record];
      }
      for (std::size_t rank = 0; rank < fitted_ranks; ++rank) {
        const std::size_t output_index =
            (state * kRank + rank) * kD + row;
        values[row] -= Float((transpose ? new_right : new_left)[output_index]) *
                       static_cast<float>(new_coefficients[rank]);
      }
    }
    double squared_norm = 0.0;
    for (float value : values) squared_norm += value * value;
    const float inverse = squared_norm > 1.0e-20
                              ? 1.0F / std::sqrt(squared_norm)
                              : 0.0F;
    for (std::size_t row = 0; row < kD; ++row) {
      result.normalized[vector_base + row * kColumns] =
          Half(values[row] * inverse);
      result.raw[vector_base + row * kColumns] = values[row];
    }
  }
  return result;
}

double RelativeRms(const std::vector<std::uint16_t>& actual,
                   const std::vector<std::uint16_t>& expected) {
  double error = 0.0;
  double energy = 0.0;
  for (std::size_t state = 0; state < kStates; ++state) {
    for (std::size_t row = 0; row < kD; ++row) {
      const std::size_t index = (state * kD + row) * kColumns;
      const double difference = Float(actual[index]) - Float(expected[index]);
      error += difference * difference;
      energy += static_cast<double>(Float(expected[index])) *
                Float(expected[index]);
    }
  }
  return std::sqrt(error / energy);
}

double RelativeRmsFlat(const std::vector<std::uint16_t>& actual,
                       const std::vector<std::uint16_t>& expected) {
  double error = 0.0;
  double energy = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double difference = Float(actual[index]) - Float(expected[index]);
    error += difference * difference;
    energy += static_cast<double>(Float(expected[index])) *
              Float(expected[index]);
  }
  return std::sqrt(error / energy);
}

double RelativeRmsFloat(const std::vector<float>& actual,
                        const std::vector<float>& expected) {
  double error = 0.0;
  double energy = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double difference = actual[index] - expected[index];
    error += difference * difference;
    energy += static_cast<double>(expected[index]) * expected[index];
  }
  return std::sqrt(error / energy);
}

double RankProductRelativeRms(const std::vector<std::uint16_t>& actual_left,
                              const std::vector<std::uint16_t>& actual_right,
                              const std::vector<std::uint16_t>& expected_left,
                              const std::vector<std::uint16_t>& expected_right) {
  double error = 0.0;
  double energy = 0.0;
  for (std::size_t state = 0; state < kStates; ++state) {
    for (std::size_t row = 0; row < kD; ++row) {
      for (std::size_t column = 0; column < kD; ++column) {
        double actual = 0.0;
        double expected = 0.0;
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          actual += Float(actual_left[(state * kRank + rank) * kD + row]) *
                    Float(actual_right[(state * kRank + rank) * kD + column]);
          expected += Float(expected_left[(state * kRank + rank) * kD + row]) *
                      Float(expected_right[(state * kRank + rank) * kD + column]);
        }
        const double difference = actual - expected;
        error += difference * difference;
        energy += expected * expected;
      }
    }
  }
  return energy > 1.0e-30 ? std::sqrt(error / energy) : std::sqrt(error);
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
    if (argc != 2 && argc != 3)
      throw std::invalid_argument("usage: probe DEVICE [MODEL_FIXTURE_DIR]");
    const bool model_fixture = argc == 3;
    const std::filesystem::path fixture_directory =
        model_fixture ? std::filesystem::path(argv[2]) : std::filesystem::path();
    device = std::stoi(argv[1]);
    Check(aclInit(nullptr), "aclInit"); initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice"); device_set = true;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    auto* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B2");
    if (platform == nullptr) throw std::runtime_error("platform unavailable");

    std::vector<std::int8_t> residual(kStates * kD * kD);
    std::vector<float> smoothing(kStates * kD), scales(kStates * kD);
    std::vector<std::uint16_t> old_left(kStates * kD * kRank);
    std::vector<std::uint16_t> old_right(old_left.size());
    std::vector<float> record_decay(kStates * kWindow);
    std::vector<std::uint16_t> record_left(kStates * kWindow * kD);
    std::vector<std::uint16_t> record_right(record_left.size());
    std::vector<std::uint16_t> input(kStates * kD * kColumns, Half(0.0F));
    for (std::size_t state = 0; state < kStates; ++state) {
      double norm = 0.0;
      for (std::size_t row = 0; row < kD; ++row) {
        smoothing[state * kD + row] = 0.35F + 0.006F * ((state + row) % 31);
        scales[state * kD + row] = 0.45F + 0.004F * ((state * 3 + row) % 37);
        const float value = static_cast<float>(
            static_cast<int>((state + 2) * (row + 7) % 29) - 14);
        input[(state * kD + row) * kColumns] = Half(value);
        norm += value * value;
        for (std::size_t column = 0; column < kD; ++column) {
          residual[(state * kD + row) * kD + column] =
              static_cast<std::int8_t>(static_cast<int>(
                  (state * 19 + row * 7 + column * 13 + 3) % 127) - 63);
        }
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          old_left[(state * kD + row) * kRank + rank] = Half(
              0.012F * std::sin((state * 11 + row * 3 + rank + 1) * 0.07F));
          old_right[(state * kD + row) * kRank + rank] = Half(
              0.011F * std::cos((state * 5 + row * 7 + rank + 2) * 0.05F));
        }
      }
      const float inverse = 1.0F / std::sqrt(norm);
      for (std::size_t row = 0; row < kD; ++row) {
        const std::size_t index = (state * kD + row) * kColumns;
        input[index] = Half(Float(input[index]) * inverse);
      }
      for (std::size_t record = 0; record < kWindow; ++record) {
        record_decay[state * kWindow + record] =
            0.965F + 0.001F * ((state + record) % 11);
        for (std::size_t axis = 0; axis < kD; ++axis) {
          const std::size_t index = (state * kWindow + record) * kD + axis;
          record_left[index] = Half(0.004F * std::sin(
              (state * 13 + record * 5 + axis + 1) * 0.03F));
          record_right[index] = Half(0.004F * std::cos(
              (state * 7 + record * 3 + axis + 2) * 0.04F));
        }
      }
    }
    std::vector<float> cycle_record_decay(kStates * kWindow, 0.0F);
    std::vector<std::uint16_t> cycle_record_left(
        kStates * kWindow * kD, Half(0.0F));
    std::vector<std::uint16_t> cycle_record_right(
        kStates * kWindow * kD, Half(0.0F));
    std::vector<std::uint32_t> cycle_positions(kStates * 8, 0U);
    std::vector<float> decay(kWindow * kStates);
    std::vector<std::uint16_t> write_keys(kWindow * kStates * kD);
    std::vector<std::uint16_t> reads(kWindow * kStates * kD);
    std::vector<std::uint16_t> queries(kWindow * kStates * kD);
    std::vector<std::uint16_t> values(kWindow * kStates * kD);
    std::vector<float> initial_dense(kStates * kD * kD, 0.0F);
    for (std::size_t state = 0; state < kStates; ++state) {
      for (std::size_t step = 0; step < kWindow; ++step) {
        decay[step * kStates + state] = 0.955F + 0.001F * (state % 7);
        for (std::size_t axis = 0; axis < kD; ++axis) {
          const std::size_t vector_index =
              (step * kStates + state) * kD + axis;
          write_keys[vector_index] = Half(0.035F * std::sin(
              static_cast<float>(state * kD + axis + 1) * 0.013F));
          reads[vector_index] = Half(0.031F * std::cos(
              static_cast<float>(state * kD + axis + 3) * 0.017F));
          queries[vector_index] = Half(0.029F * std::sin(
              static_cast<float>(state * kD + axis + 5) * 0.019F));
          values[vector_index] = Half(0.12F * std::cos(
              static_cast<float>(state * kD + axis + 7) * 0.011F));
        }
      }
      for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t column = 0; column < kD; ++column) {
          const std::size_t cell = (state * kD + row) * kD + column;
          float dense = static_cast<float>(residual[cell]) / 127.0F *
                        smoothing[state * kD + row] *
                        scales[state * kD + column];
          for (std::size_t rank = 0; rank < kRank; ++rank) {
            dense += Float(old_left[(state * kD + row) * kRank + rank]) *
                     Float(old_right[(state * kD + column) * kRank + rank]);
          }
          initial_dense[cell] = dense;
        }
      }
    }
    if (model_fixture) {
      residual = ReadBinary<std::int8_t>(
          fixture_directory / "residual.i8", residual.size());
      smoothing = ReadBinary<float>(fixture_directory / "smoothing.f32",
                                    smoothing.size());
      scales = ReadBinary<float>(fixture_directory / "scales.f32",
                                 scales.size());
      old_left = ReadBinary<std::uint16_t>(
          fixture_directory / "compensator_keys.f16", old_left.size());
      old_right = ReadBinary<std::uint16_t>(
          fixture_directory / "compensator_values.f16", old_right.size());
      initial_dense = ReadBinary<float>(fixture_directory / "initial_dense.f32",
                                        initial_dense.size());
      decay = ReadBinary<float>(fixture_directory / "decay.f32", decay.size());
      write_keys = ReadBinary<std::uint16_t>(
          fixture_directory / "write_keys.f16", write_keys.size());
      reads = ReadBinary<std::uint16_t>(fixture_directory / "reads.f16",
                                        reads.size());
      queries = ReadBinary<std::uint16_t>(fixture_directory / "queries.f16",
                                          queries.size());
      values = ReadBinary<std::uint16_t>(fixture_directory / "values.f16",
                                         values.size());
      // Empty record slots contribute no rank-one term, but their decay is the
      // multiplicative identity until the window kernel writes a real record.
      std::fill(record_decay.begin(), record_decay.end(), 1.0F);
      std::fill(record_left.begin(), record_left.end(), Half(0.0F));
      std::fill(record_right.begin(), record_right.end(), Half(0.0F));
    }
    std::vector<std::uint16_t> zero_basis(kStates * kRank * kD,
                                          Half(0.0F));
    const auto expected_forward_result = Project(
        residual, smoothing, scales, old_left, old_right, record_decay,
        record_left, record_right, zero_basis, zero_basis, input, 0, false,
        true);
    const auto exact_forward_result = Project(
        residual, smoothing, scales, old_left, old_right, record_decay,
        record_left, record_right, zero_basis, zero_basis, input, 0, false,
        false);
    const auto expected_transpose_result = Project(
        residual, smoothing, scales, old_left, old_right, record_decay,
        record_left, record_right, zero_basis, zero_basis, input, 0, true,
        true);
    const auto exact_transpose_result = Project(
        residual, smoothing, scales, old_left, old_right, record_decay,
        record_left, record_right, zero_basis, zero_basis, input, 0, true,
        false);
    const auto& expected_forward = expected_forward_result.normalized;
    const auto& exact_forward = exact_forward_result.normalized;
    const auto& expected_transpose = expected_transpose_result.normalized;
    const auto& exact_transpose = exact_transpose_result.normalized;
    constexpr std::uint32_t kPowerIterations = 8;
    const std::size_t vector_elements = kStates * kD * kColumns;
    std::vector<std::uint16_t> seeds(kRank * vector_elements, Half(0.0F));
    for (std::size_t rank = 0; rank < kRank; ++rank) {
      for (std::size_t state = 0; state < kStates; ++state) {
        double squared_norm = 0.0;
        for (std::size_t row = 0; row < kD; ++row) {
          const float value = static_cast<float>(static_cast<int>(
              (state + rank + 2) * (row + rank * 5 + 3) % 23) -
                                                11);
          seeds[rank * vector_elements +
                (state * kD + row) * kColumns] = Half(value);
          squared_norm += value * value;
        }
        const float inverse = 1.0F / std::sqrt(squared_norm);
        for (std::size_t row = 0; row < kD; ++row) {
          const std::size_t index =
              rank * vector_elements + (state * kD + row) * kColumns;
          seeds[index] = Half(Float(seeds[index]) * inverse);
        }
      }
    }
    auto reference_new_left = zero_basis;
    auto reference_new_right = zero_basis;
    for (std::size_t rank = 0; rank < kRank; ++rank) {
      std::vector<std::uint16_t> right(
          seeds.begin() + static_cast<std::ptrdiff_t>(rank * vector_elements),
          seeds.begin() +
              static_cast<std::ptrdiff_t>((rank + 1) * vector_elements));
      for (std::uint32_t iteration = 0; iteration < kPowerIterations;
           ++iteration) {
        const auto left = Project(
            residual, smoothing, scales, old_left, old_right, record_decay,
            record_left, record_right, reference_new_left,
            reference_new_right, right, rank, false, true);
        const auto transposed = Project(
            residual, smoothing, scales, old_left, old_right, record_decay,
            record_left, record_right, reference_new_left,
            reference_new_right, left.normalized, rank, true, true);
        right = transposed.normalized;
      }
      const auto final_projection = Project(
          residual, smoothing, scales, old_left, old_right, record_decay,
          record_left, record_right, reference_new_left, reference_new_right,
          right, rank, false, true);
      for (std::size_t state = 0; state < kStates; ++state) {
        for (std::size_t row = 0; row < kD; ++row) {
          const std::size_t vector_index =
              (state * kD + row) * kColumns;
          const std::size_t basis_index =
              (state * kRank + rank) * kD + row;
          reference_new_left[basis_index] =
              Half(final_projection.raw[vector_index]);
          reference_new_right[basis_index] = right[vector_index];
        }
      }
    }

#define DEVICE_BUFFER(name, source) \
    DeviceBuffer<typename decltype(source)::value_type> name(source.size()); \
    ToDevice(name, source)
    DEVICE_BUFFER(residual_device, residual);
    DEVICE_BUFFER(smoothing_device, smoothing);
    DEVICE_BUFFER(scales_device, scales);
    DEVICE_BUFFER(old_left_device, old_left);
    DEVICE_BUFFER(old_right_device, old_right);
    DEVICE_BUFFER(record_decay_device, record_decay);
    DEVICE_BUFFER(record_left_device, record_left);
    DEVICE_BUFFER(record_right_device, record_right);
    DEVICE_BUFFER(input_device, input);
    DEVICE_BUFFER(seeds_device, seeds);
    DEVICE_BUFFER(new_left_device, zero_basis);
    DEVICE_BUFFER(new_right_device, zero_basis);
    DEVICE_BUFFER(positions_device, cycle_positions);
    DEVICE_BUFFER(decay_device, decay);
    DEVICE_BUFFER(write_keys_device, write_keys);
    DEVICE_BUFFER(reads_device, reads);
    DEVICE_BUFFER(queries_device, queries);
    DEVICE_BUFFER(values_device, values);
    DEVICE_BUFFER(dense_device, initial_dense);
#undef DEVICE_BUFFER
    DeviceBuffer<std::int8_t> quantized_device(input.size());
    DeviceBuffer<std::int32_t> projected_device(input.size());
    DeviceBuffer<float> dynamic_scale_device(kStates * 8);
    DeviceBuffer<std::uint16_t> output_device(input.size());
    DeviceBuffer<float> corrected_device(input.size());
    DeviceBuffer<std::int8_t> next_residual_device(residual.size());
    DeviceBuffer<float> next_smoothing_device(smoothing.size());
    DeviceBuffer<float> next_scales_device(scales.size());
    DeviceBuffer<std::uint16_t> next_left_device(old_left.size());
    DeviceBuffer<std::uint16_t> next_right_device(old_right.size());
    DeviceBuffer<float> residual_workspace_device(residual.size());
    DeviceBuffer<float> window_output_device(kStates * kD);
    DeviceBuffer<float> cube_output_device(kStates * kD);
    DeviceBuffer<float> fp32_output_device(kStates * kD);
    const auto forward_tiling = MakeTiling(*platform, false);
    const auto transpose_tiling = MakeTiling(*platform, true);
    DeviceBuffer<std::uint8_t> forward_tiling_device(forward_tiling.size());
    DeviceBuffer<std::uint8_t> transpose_tiling_device(transpose_tiling.size());
    ToDevice(forward_tiling_device, forward_tiling);
    ToDevice(transpose_tiling_device, transpose_tiling);
    DeviceBuffer<std::uint8_t> workspace_device(
        platform->GetLibApiWorkSpaceSize());
    constexpr std::uint32_t kBlocks = (kStates + 1U) / 2U;
    auto launch = [&](const DeviceBuffer<std::uint16_t>& projection_input,
                      DeviceBuffer<std::uint16_t>& projection_output,
                      bool transpose, std::uint32_t fitted_ranks) {
      const auto& axis_scale = transpose ? smoothing_device : scales_device;
      if (statecentric_gdn_state_codec_prepare_int8_vector_launch_v1(
              stream, projection_input.data(), axis_scale.data(),
              quantized_device.data(), dynamic_scale_device.data(), kStates) !=
          0) throw std::runtime_error("prepare admission failed");
      if (statecentric_gdn_state_codec_int8_cube_matmul_launch_v1(
              stream, residual_device.data(), quantized_device.data(),
              projected_device.data(), kStates, transpose ? 1U : 0U,
              workspace_device.data(),
              transpose ? transpose_tiling_device.data()
                        : forward_tiling_device.data(),
              kBlocks) != 0) throw std::runtime_error("Cube admission failed");
      if (statecentric_gdn_state_codec_finish_encoded_projection_launch_v1(
              stream, projected_device.data(), dynamic_scale_device.data(),
              projection_input.data(), smoothing_device.data(),
              scales_device.data(), old_left_device.data(), old_right_device.data(),
              record_decay_device.data(), record_left_device.data(),
              record_right_device.data(), new_left_device.data(),
              new_right_device.data(), projection_output.data(), kStates,
              fitted_ranks, transpose ? 1U : 0U) != 0)
        throw std::runtime_error("finish admission failed");
    };
    launch(input_device, output_device, false, 0);
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    std::vector<double> timings;
    for (int repetition = 0; repetition < 11; ++repetition) {
      const auto start = std::chrono::steady_clock::now();
      launch(input_device, output_device, false, 0);
      Check(aclrtSynchronizeStream(stream), "timed synchronize");
      const auto stop = std::chrono::steady_clock::now();
      timings.push_back(std::chrono::duration<double, std::micro>(stop - start)
                            .count());
    }
    const auto actual_forward = FromDevice(output_device);
    const auto forward_quantized = FromDevice(quantized_device);
    const auto forward_projected = FromDevice(projected_device);
    const auto forward_dynamic_scale = FromDevice(dynamic_scale_device);
    launch(input_device, output_device, true, 0);
    Check(aclrtSynchronizeStream(stream), "transpose synchronize");
    const auto actual_transpose = FromDevice(output_device);
    auto rank_chain = [&] {
      for (std::uint32_t rank = 0; rank < kRank; ++rank) {
        Check(aclrtMemcpyAsync(
                  input_device.data(), input_device.bytes(),
                  seeds_device.data() + rank * vector_elements,
                  input_device.bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "rank seed D2D");
        for (std::uint32_t iteration = 0; iteration < kPowerIterations;
             ++iteration) {
          launch(input_device, output_device, false, rank);
          launch(output_device, input_device, true, rank);
        }
        const auto& axis_scale = scales_device;
        if (statecentric_gdn_state_codec_prepare_int8_vector_launch_v1(
                stream, input_device.data(), axis_scale.data(),
                quantized_device.data(), dynamic_scale_device.data(),
                kStates) != 0)
          throw std::runtime_error("commit prepare admission failed");
        if (statecentric_gdn_state_codec_int8_cube_matmul_launch_v1(
                stream, residual_device.data(), quantized_device.data(),
                projected_device.data(), kStates, 0U, workspace_device.data(),
                forward_tiling_device.data(), kBlocks) != 0)
          throw std::runtime_error("commit Cube admission failed");
        if (statecentric_gdn_state_codec_commit_encoded_rank_launch_v1(
                stream, projected_device.data(), dynamic_scale_device.data(),
                input_device.data(), smoothing_device.data(),
                scales_device.data(), old_left_device.data(),
                old_right_device.data(), record_decay_device.data(),
                record_left_device.data(), record_right_device.data(),
                new_left_device.data(), new_right_device.data(),
                corrected_device.data(), kStates, rank) != 0)
          throw std::runtime_error("commit rank admission failed");
      }
    };
    auto reset_bases = [&] {
      Check(aclrtMemsetAsync(new_left_device.data(), new_left_device.bytes(), 0,
                             new_left_device.bytes(), stream),
            "left basis memset");
      Check(aclrtMemsetAsync(new_right_device.data(), new_right_device.bytes(),
                             0, new_right_device.bytes(), stream),
            "right basis memset");
    };
    reset_bases();
    rank_chain();
    Check(aclrtSynchronizeStream(stream), "rank chain warmup synchronize");
    std::vector<double> rank_timings;
    for (int repetition = 0; repetition < 11; ++repetition) {
      reset_bases();
      const auto start = std::chrono::steady_clock::now();
      rank_chain();
      Check(aclrtSynchronizeStream(stream), "rank chain timed synchronize");
      const auto stop = std::chrono::steady_clock::now();
      rank_timings.push_back(
          std::chrono::duration<double, std::micro>(stop - start).count());
    }
    const auto actual_new_left = FromDevice(new_left_device);
    const auto actual_new_right = FromDevice(new_right_device);
    const double rank_left_rms =
        RelativeRmsFlat(actual_new_left, reference_new_left);
    const double rank_right_rms =
        RelativeRmsFlat(actual_new_right, reference_new_right);
    const double rank_product_rms = RankProductRelativeRms(
        actual_new_left, actual_new_right, reference_new_left,
        reference_new_right);
    auto requantize = [&] {
      if (statecentric_gdn_state_codec_encoded_requantize_launch_v1(
              stream, residual_device.data(), smoothing_device.data(),
              scales_device.data(), old_left_device.data(),
              old_right_device.data(), record_decay_device.data(),
              record_left_device.data(), record_right_device.data(),
              new_left_device.data(), new_right_device.data(),
              next_residual_device.data(), next_smoothing_device.data(),
              next_scales_device.data(), next_left_device.data(),
              next_right_device.data(), residual_workspace_device.data(),
              kStates) != 0)
        throw std::runtime_error("requantize admission failed");
    };
    requantize();
    Check(aclrtSynchronizeStream(stream), "requantize warmup synchronize");
    std::vector<double> requantize_timings;
    for (int repetition = 0; repetition < 11; ++repetition) {
      const auto start = std::chrono::steady_clock::now();
      requantize();
      Check(aclrtSynchronizeStream(stream), "requantize timed synchronize");
      const auto stop = std::chrono::steady_clock::now();
      requantize_timings.push_back(
          std::chrono::duration<double, std::micro>(stop - start).count());
    }
    std::vector<double> boundary_timings;
    for (int repetition = 0; repetition < 11; ++repetition) {
      reset_bases();
      const auto start = std::chrono::steady_clock::now();
      rank_chain();
      requantize();
      Check(aclrtSynchronizeStream(stream), "boundary timed synchronize");
      const auto stop = std::chrono::steady_clock::now();
      boundary_timings.push_back(
          std::chrono::duration<double, std::micro>(stop - start).count());
    }
    const auto next_residual = FromDevice(next_residual_device);
    const auto next_smoothing = FromDevice(next_smoothing_device);
    const auto next_scales = FromDevice(next_scales_device);
    const auto next_left = FromDevice(next_left_device);
    const auto next_right = FromDevice(next_right_device);
    const auto final_new_left = FromDevice(new_left_device);
    const auto final_new_right = FromDevice(new_right_device);
    double reconstruction_error = 0.0;
    double reconstruction_energy = 0.0;
    double factor_copy_error = 0.0;
    bool finite_scales = true;
    for (std::size_t state = 0; state < kStates; ++state) {
      std::vector<float> record_suffix(kWindow);
      float suffix = 1.0F;
      for (std::size_t reverse = kWindow; reverse > 0; --reverse) {
        const std::size_t record = reverse - 1;
        record_suffix[record] = suffix;
        suffix *= record_decay[state * kWindow + record];
      }
      for (std::size_t axis = 0; axis < kD; ++axis) {
        finite_scales = finite_scales &&
                        std::isfinite(next_smoothing[state * kD + axis]) &&
                        std::isfinite(next_scales[state * kD + axis]);
        for (std::size_t rank = 0; rank < kRank; ++rank) {
          const std::size_t interleaved =
              (state * kD + axis) * kRank + rank;
          const std::size_t rank_major =
              (state * kRank + rank) * kD + axis;
          factor_copy_error = std::max(
              factor_copy_error,
              std::abs(static_cast<double>(Float(next_left[interleaved]) -
                                           Float(final_new_left[rank_major]))));
          factor_copy_error = std::max(
              factor_copy_error,
              std::abs(static_cast<double>(Float(next_right[interleaved]) -
                                           Float(final_new_right[rank_major]))));
        }
      }
      for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t column = 0; column < kD; ++column) {
          const std::size_t cell = (state * kD + row) * kD + column;
          double original = static_cast<float>(residual[cell]) / 127.0F *
                            smoothing[state * kD + row] *
                            scales[state * kD + column] * suffix;
          for (std::size_t rank = 0; rank < kRank; ++rank) {
            original +=
                Float(old_left[(state * kD + row) * kRank + rank]) *
                Float(old_right[(state * kD + column) * kRank + rank]) *
                suffix;
          }
          for (std::size_t record = 0; record < kWindow; ++record) {
            original +=
                Float(record_left[(state * kWindow + record) * kD + row]) *
                Float(record_right[(state * kWindow + record) * kD + column]) *
                record_suffix[record];
          }
          double reconstructed =
              static_cast<float>(next_residual[cell]) / 127.0F *
              next_smoothing[state * kD + row] *
              next_scales[state * kD + column];
          for (std::size_t rank = 0; rank < kRank; ++rank) {
            reconstructed +=
                Float(next_left[(state * kD + row) * kRank + rank]) *
                Float(next_right[(state * kD + column) * kRank + rank]);
          }
          const double difference = reconstructed - original;
          reconstruction_error += difference * difference;
          reconstruction_energy += original * original;
        }
      }
    }
    const double reconstruction_rms =
        std::sqrt(reconstruction_error / reconstruction_energy);
    auto restore_cycle = [&] {
      ToDevice(record_decay_device, cycle_record_decay);
      ToDevice(record_left_device, cycle_record_left);
      ToDevice(record_right_device, cycle_record_right);
      ToDevice(positions_device, cycle_positions);
    };
    auto restore_dense = [&] { ToDevice(dense_device, initial_dense); };
    auto launch_window_step = [&](std::size_t step) {
      const std::size_t vector_offset = step * kStates * kD;
      const std::size_t scalar_offset = step * kStates;
      if (statecentric_gdn_state_codec_windowed_gdn_step_launch_v1(
              stream, residual_device.data(), smoothing_device.data(),
              scales_device.data(), old_left_device.data(),
              old_right_device.data(), record_decay_device.data(),
              record_left_device.data(), record_right_device.data(),
              positions_device.data(), decay_device.data() + scalar_offset,
              write_keys_device.data() + vector_offset,
              reads_device.data() + vector_offset,
              queries_device.data() + vector_offset,
              values_device.data() + vector_offset,
              window_output_device.data(), kStates) != 0)
        throw std::runtime_error("window step admission failed");
    };
    auto launch_cube_step = [&](std::size_t step) {
      const std::size_t vector_offset = step * kStates * kD;
      const std::size_t scalar_offset = step * kStates;
      if (statecentric_gdn_state_codec_prepare_decode_pair_launch_v1(
              stream, reads_device.data() + vector_offset,
              queries_device.data() + vector_offset,
              decay_device.data() + scalar_offset, smoothing_device.data(),
              quantized_device.data(), dynamic_scale_device.data(), kStates) != 0)
        throw std::runtime_error("decode pair prepare admission failed");
      if (statecentric_gdn_state_codec_int8_cube_matmul_launch_v1(
              stream, residual_device.data(), quantized_device.data(),
              projected_device.data(), kStates, 1U, workspace_device.data(),
              transpose_tiling_device.data(), kBlocks) != 0)
        throw std::runtime_error("decode pair Cube admission failed");
      if (statecentric_gdn_state_codec_finish_decode_pair_launch_v1(
              stream, projected_device.data(), dynamic_scale_device.data(),
              scales_device.data(), old_left_device.data(),
              old_right_device.data(), record_decay_device.data(),
              record_left_device.data(), record_right_device.data(),
              positions_device.data(), decay_device.data() + scalar_offset,
              write_keys_device.data() + vector_offset,
              reads_device.data() + vector_offset,
              queries_device.data() + vector_offset,
              values_device.data() + vector_offset, cube_output_device.data(),
              kStates) != 0)
        throw std::runtime_error("decode pair finish admission failed");
    };
    auto launch_fp32_step = [&](std::size_t step) {
      const std::size_t vector_offset = step * kStates * kD;
      const std::size_t scalar_offset = step * kStates;
      if (statecentric_gdn_state_codec_fp32_gdn_step_launch_v1(
              stream, dense_device.data(), decay_device.data() + scalar_offset,
              write_keys_device.data() + vector_offset,
              reads_device.data() + vector_offset,
              queries_device.data() + vector_offset,
              values_device.data() + vector_offset,
              fp32_output_device.data(), kStates) != 0)
        throw std::runtime_error("FP32 step admission failed");
    };
    restore_cycle();
    launch_cube_step(0);
    Check(aclrtSynchronizeStream(stream), "first Cube step synchronize");
    const auto first_cube_output = FromDevice(cube_output_device);
    const auto first_pair_scales = FromDevice(dynamic_scale_device);
    const auto first_pair_projected = FromDevice(projected_device);
    const auto first_pair_quantized = FromDevice(quantized_device);
    restore_cycle();
    launch_window_step(0);
    Check(aclrtSynchronizeStream(stream), "first scalar step synchronize");
    const auto first_window_output = FromDevice(window_output_device);
    restore_cycle();
    for (std::size_t step = 0; step < kWindow; ++step)
      launch_cube_step(step);
    Check(aclrtSynchronizeStream(stream), "Cube step exactness synchronize");
    const auto cycle_cube_output = FromDevice(cube_output_device);
    restore_cycle();
    restore_dense();
    reset_bases();
    for (std::size_t step = 0; step < kWindow; ++step) {
      launch_window_step(step);
      launch_fp32_step(step);
    }
    rank_chain();
    requantize();
    Check(aclrtSynchronizeStream(stream), "cycle exactness synchronize");
    const auto cycle_window_output = FromDevice(window_output_device);
    const auto cycle_fp32_output = FromDevice(fp32_output_device);
    const auto cycle_fp32_dense = FromDevice(dense_device);
    const auto cycle_residual = FromDevice(next_residual_device);
    const auto cycle_smoothing = FromDevice(next_smoothing_device);
    const auto cycle_scales = FromDevice(next_scales_device);
    const auto cycle_left = FromDevice(next_left_device);
    const auto cycle_right = FromDevice(next_right_device);
    std::vector<float> cycle_reconstructed(cycle_fp32_dense.size(), 0.0F);
    for (std::size_t state = 0; state < kStates; ++state) {
      for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t column = 0; column < kD; ++column) {
          const std::size_t cell = (state * kD + row) * kD + column;
          float value = static_cast<float>(cycle_residual[cell]) / 127.0F *
                        cycle_smoothing[state * kD + row] *
                        cycle_scales[state * kD + column];
          for (std::size_t rank = 0; rank < kRank; ++rank) {
            value +=
                Float(cycle_left[(state * kD + row) * kRank + rank]) *
                Float(cycle_right[(state * kD + column) * kRank + rank]);
          }
          cycle_reconstructed[cell] = value;
        }
      }
    }
    const double cycle_output_rms =
        RelativeRmsFloat(cycle_window_output, cycle_fp32_output);
    const double cube_step_output_rms =
        RelativeRmsFloat(cycle_cube_output, cycle_window_output);
    const double first_cube_step_output_rms =
        RelativeRmsFloat(first_cube_output, first_window_output);
    const double cycle_state_rms =
        RelativeRmsFloat(cycle_reconstructed, cycle_fp32_dense);
    std::vector<double> compressed_cycle_timings;
    std::vector<double> fp32_cycle_timings;
    std::vector<double> cube_window_timings;
    for (int warmup = 0; warmup < 3; ++warmup) {
      restore_cycle();
      reset_bases();
      for (std::size_t step = 0; step < kWindow; ++step)
        launch_window_step(step);
      rank_chain();
      requantize();
      restore_dense();
      for (std::size_t step = 0; step < kWindow; ++step)
        launch_fp32_step(step);
    }
    Check(aclrtSynchronizeStream(stream), "cycle warmup synchronize");
    for (int repetition = 0; repetition < 11; ++repetition) {
      restore_cycle();
      Check(aclrtSynchronizeStream(stream), "Cube restore synchronize");
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t step = 0; step < kWindow; ++step)
        launch_cube_step(step);
      Check(aclrtSynchronizeStream(stream), "Cube window synchronize");
      cube_window_timings.push_back(
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - start).count());
    }
    for (int repetition = 0; repetition < 11; ++repetition) {
      if (repetition % 2 == 0) {
        restore_cycle();
        reset_bases();
        Check(aclrtSynchronizeStream(stream), "compressed restore synchronize");
        auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < kWindow; ++step)
          launch_window_step(step);
        rank_chain();
        requantize();
        Check(aclrtSynchronizeStream(stream), "compressed cycle synchronize");
        auto stop = std::chrono::steady_clock::now();
        compressed_cycle_timings.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
        restore_dense();
        start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < kWindow; ++step)
          launch_fp32_step(step);
        Check(aclrtSynchronizeStream(stream), "FP32 cycle synchronize");
        stop = std::chrono::steady_clock::now();
        fp32_cycle_timings.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
      } else {
        restore_dense();
        auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < kWindow; ++step)
          launch_fp32_step(step);
        Check(aclrtSynchronizeStream(stream), "FP32 cycle synchronize");
        auto stop = std::chrono::steady_clock::now();
        fp32_cycle_timings.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
        restore_cycle();
        reset_bases();
        Check(aclrtSynchronizeStream(stream), "compressed restore synchronize");
        start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < kWindow; ++step)
          launch_window_step(step);
        rank_chain();
        requantize();
        Check(aclrtSynchronizeStream(stream), "compressed cycle synchronize");
        stop = std::chrono::steady_clock::now();
        compressed_cycle_timings.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
      }
    }
    const double compressed_cycle_us = Median(compressed_cycle_timings);
    const double fp32_cycle_us = Median(fp32_cycle_timings);
    const double cycle_speedup = fp32_cycle_us / compressed_cycle_us;
    const double cube_window_us = Median(cube_window_timings);
    const double matched_forward = RelativeRms(actual_forward, expected_forward);
    const double matched_transpose =
        RelativeRms(actual_transpose, expected_transpose);
    const double quantization_forward =
        RelativeRms(expected_forward, exact_forward);
    const double quantization_transpose =
        RelativeRms(expected_transpose, exact_transpose);
    const bool passed = matched_forward < 3.0e-3 &&
                        matched_transpose < 3.0e-3 &&
                        quantization_forward < 2.0e-2 &&
                        quantization_transpose < 2.0e-2 &&
                        rank_left_rms < 5.0e-3 &&
                        rank_product_rms < 5.0e-3 && finite_scales &&
                        factor_copy_error == 0.0 &&
                        reconstruction_rms < 2.0e-2 &&
                        cycle_output_rms < 2.5e-3 &&
                        first_cube_step_output_rms < 2.5e-2 &&
                        cube_step_output_rms < 2.5e-2 &&
                        cycle_state_rms < 2.0e-2 &&
                        compressed_cycle_us < fp32_cycle_us;
    std::size_t worst_state = 0;
    double worst_state_error = -1.0;
    float minimum_dynamic_scale = forward_dynamic_scale[0];
    for (std::size_t state = 0; state < kStates; ++state) {
      double state_error = 0.0;
      for (std::size_t row = 0; row < kD; ++row) {
        const std::size_t index = (state * kD + row) * kColumns;
        const double difference =
            Float(actual_forward[index]) - Float(expected_forward[index]);
        state_error += difference * difference;
      }
      if (state_error > worst_state_error) {
        worst_state_error = state_error;
        worst_state = state;
      }
      minimum_dynamic_scale = std::min(
          minimum_dynamic_scale, forward_dynamic_scale[state * 8]);
    }
    const std::size_t worst_index = worst_state * kD * kColumns;
    std::cout << "evidence_label="
              << (model_fixture
                      ? "model-derived-qwen35-layer0-agentx-first-turn-gate"
                      : "synthetic-encoded-int8-rank4-gate")
              << '\n'
              << "fixture_directory="
              << (model_fixture ? fixture_directory.string() : "none") << '\n'
              << "states=" << kStates << '\n'
              << "matched_forward_relative_rms=" << matched_forward << '\n'
              << "matched_transpose_relative_rms=" << matched_transpose << '\n'
              << "vector_quantization_forward_relative_rms="
              << quantization_forward << '\n'
              << "vector_quantization_transpose_relative_rms="
              << quantization_transpose << '\n'
              << "median_forward_chain_us=" << Median(timings) << '\n'
              << "rank4_left_relative_rms=" << rank_left_rms << '\n'
              << "rank4_right_relative_rms=" << rank_right_rms << '\n'
              << "rank4_product_relative_rms=" << rank_product_rms << '\n'
              << "rank4_median_chain_us=" << Median(rank_timings) << '\n'
              << "requantize_median_us=" << Median(requantize_timings) << '\n'
              << "boundary_median_us=" << Median(boundary_timings) << '\n'
              << "reconstruction_relative_rms=" << reconstruction_rms << '\n'
              << "factor_copy_max_abs=" << factor_copy_error << '\n'
              << "cycle_output_relative_rms=" << cycle_output_rms << '\n'
              << "first_cube_step_output_relative_rms="
              << first_cube_step_output_rms << '\n'
              << "cube_step_output_relative_rms=" << cube_step_output_rms << '\n'
              << "cube_window_median_us=" << cube_window_us << '\n'
              << "debug_first_cube_output0=" << first_cube_output[0] << '\n'
              << "debug_first_scalar_output0=" << first_window_output[0] << '\n'
              << "debug_pair_scales0=" << first_pair_scales[0] << ','
              << first_pair_scales[1] << ',' << first_pair_scales[2] << ','
              << first_pair_scales[3] << '\n'
              << "debug_pair_projected0=" << first_pair_projected[0] << ','
              << first_pair_projected[1] << ',' << first_pair_projected[2]
              << ',' << first_pair_projected[3] << '\n'
              << "debug_pair_quantized0="
              << static_cast<int>(first_pair_quantized[0]) << ','
              << static_cast<int>(first_pair_quantized[1]) << ','
              << static_cast<int>(first_pair_quantized[2]) << ','
              << static_cast<int>(first_pair_quantized[3]) << '\n'
              << "cycle_state_relative_rms=" << cycle_state_rms << '\n'
              << "compressed_cycle_median_us=" << compressed_cycle_us << '\n'
              << "fp32_cycle_median_us=" << fp32_cycle_us << '\n'
              << "cycle_speedup=" << cycle_speedup << '\n'
              << "debug_forward_actual0=" << Float(actual_forward[0]) << '\n'
              << "debug_forward_expected0=" << Float(expected_forward[0]) << '\n'
              << "debug_forward_exact0=" << Float(exact_forward[0]) << '\n'
              << "debug_forward_dynamic_scale0=" << forward_dynamic_scale[0]
              << '\n'
              << "debug_forward_quantized0="
              << static_cast<int>(forward_quantized[0]) << '\n'
              << "debug_forward_projected0=" << forward_projected[0] << '\n'
              << "debug_minimum_dynamic_scale=" << minimum_dynamic_scale << '\n'
              << "debug_worst_state=" << worst_state << '\n'
              << "debug_worst_actual0=" << Float(actual_forward[worst_index])
              << '\n'
              << "debug_worst_expected0="
              << Float(expected_forward[worst_index]) << '\n'
              << "debug_worst_dynamic_scale="
              << forward_dynamic_scale[worst_state * 8] << '\n'
              << "gate=" << (passed ? "pass" : "fail") << '\n';
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream"); stream = nullptr;
    Check(aclrtResetDevice(device), "aclrtResetDevice"); device_set = false;
    Check(aclFinalize(), "aclFinalize"); initialized = false;
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(device);
    if (initialized) (void)aclFinalize();
    return 1;
  }
}

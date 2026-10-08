#include "statecentric/leapquant_state_capsule.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "statecentric/sha256.h"

namespace statecentric {
namespace {

constexpr std::string_view kMagic = "LQCAPS01";
constexpr std::size_t kDigestCharacters = 64;
constexpr float kQuantizedMaximum = 127.0F;

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

void ValidateParameters(const LeapQuantParameters& parameters) {
  Require(parameters.schema_version == 1,
          "LeapQuant capsule schema version is unsupported");
  Require(parameters.window_size == 16,
          "LeapQuant capsule schema v1 fixes p=16");
  Require(parameters.compensator_rank == 4,
          "LeapQuant capsule schema v1 fixes r=4");
  Require(parameters.quant_bits == 8,
          "LeapQuant reference supports only symmetric INT8");
  Require(std::isfinite(parameters.smoothing_floor) &&
              parameters.smoothing_floor > 0.0F,
          "LeapQuant smoothing floor must be finite and positive");
}

void ValidateIdentity(const LeapQuantIdentity& identity) {
  Require(IsLowercaseSha256Hex(identity.model_geometry_sha256),
          "LeapQuant model geometry digest is invalid");
  Require(identity.generation > 0,
          "LeapQuant generation must be positive");
  Require(identity.epoch > 0, "LeapQuant epoch must be positive");
  Require(identity.key_dimension > 0 && identity.value_dimension > 0,
          "LeapQuant geometry must be nonzero");
}

float Norm(std::span<const float> values) {
  double sum = 0.0;
  for (const float value : values) sum += static_cast<double>(value) * value;
  return static_cast<float>(std::sqrt(sum));
}

bool AllFinite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

bool Normalize(std::vector<float>& values) {
  const float norm = Norm(values);
  if (!(norm > std::numeric_limits<float>::epsilon())) return false;
  for (float& value : values) value /= norm;
  return true;
}

template <typename Integer>
void AppendInteger(std::vector<std::uint8_t>& output, Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
  }
}

void AppendFloat(std::vector<std::uint8_t>& output, float value) {
  AppendInteger(output, std::bit_cast<std::uint32_t>(value));
}

template <typename Integer>
Integer ReadInteger(std::span<const std::uint8_t> input, std::size_t& offset) {
  static_assert(std::is_unsigned_v<Integer>);
  if (offset > input.size() || input.size() - offset < sizeof(Integer)) {
    throw std::invalid_argument("LeapQuant capsule is truncated");
  }
  Integer value = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    value |= static_cast<Integer>(input[offset++]) << (byte * 8U);
  }
  return value;
}

float ReadFloat(std::span<const std::uint8_t> input, std::size_t& offset) {
  return std::bit_cast<float>(ReadInteger<std::uint32_t>(input, offset));
}

void AppendFloats(std::vector<std::uint8_t>& output,
                  std::span<const float> values) {
  for (const float value : values) AppendFloat(output, value);
}

std::vector<float> ReadFloats(std::span<const std::uint8_t> input,
                              std::size_t& offset, std::size_t count) {
  if (count > (input.size() - std::min(offset, input.size())) / sizeof(float)) {
    throw std::invalid_argument("LeapQuant capsule float payload is truncated");
  }
  std::vector<float> values(count);
  for (float& value : values) value = ReadFloat(input, offset);
  return values;
}

std::string BytesSha256(std::span<const std::uint8_t> bytes) {
  return Sha256Hex(std::string_view(
      reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

}  // namespace

struct LeapQuantCapsule::Boundary {
  std::vector<std::int8_t> residual;
  std::vector<float> smoothing;
  std::vector<float> value_scales;
  std::vector<float> compensator_keys;
  std::vector<float> compensator_values;
};

namespace {

std::shared_ptr<const LeapQuantCapsule::Boundary> QuantizeBoundary(
    std::span<const float> dense, const LeapQuantIdentity& identity,
    const LeapQuantParameters& parameters) {
  const std::size_t keys = identity.key_dimension;
  const std::size_t values = identity.value_dimension;
  std::vector<float> residual(dense.begin(), dense.end());
  auto boundary = std::make_shared<LeapQuantCapsule::Boundary>();
  boundary->compensator_keys.assign(keys * parameters.compensator_rank, 0.0F);
  boundary->compensator_values.assign(values * parameters.compensator_rank,
                                      0.0F);

  for (std::size_t rank = 0; rank < parameters.compensator_rank; ++rank) {
    std::vector<float> right(values);
    for (std::size_t column = 0; column < values; ++column) {
      right[column] = std::sin(static_cast<float>(
          (rank + 1) * (column + 1)));
    }
    if (!Normalize(right)) right.front() = 1.0F;
    std::vector<float> left(keys);
    for (int iteration = 0; iteration < 32; ++iteration) {
      std::fill(left.begin(), left.end(), 0.0F);
      for (std::size_t row = 0; row < keys; ++row) {
        for (std::size_t column = 0; column < values; ++column) {
          left[row] += residual[row * values + column] * right[column];
        }
      }
      if (!Normalize(left)) break;
      std::fill(right.begin(), right.end(), 0.0F);
      for (std::size_t column = 0; column < values; ++column) {
        for (std::size_t row = 0; row < keys; ++row) {
          right[column] += residual[row * values + column] * left[row];
        }
      }
      if (!Normalize(right)) break;
    }
    float singular_value = 0.0F;
    for (std::size_t row = 0; row < keys; ++row) {
      for (std::size_t column = 0; column < values; ++column) {
        singular_value += left[row] * residual[row * values + column] *
                          right[column];
      }
    }
    if (!std::isfinite(singular_value)) {
      throw std::invalid_argument("LeapQuant compensator fit is non-finite");
    }
    for (std::size_t row = 0; row < keys; ++row) {
      boundary->compensator_keys[row * parameters.compensator_rank + rank] =
          left[row] * singular_value;
    }
    for (std::size_t column = 0; column < values; ++column) {
      boundary->compensator_values
          [column * parameters.compensator_rank + rank] = right[column];
    }
    for (std::size_t row = 0; row < keys; ++row) {
      for (std::size_t column = 0; column < values; ++column) {
        residual[row * values + column] -=
            left[row] * singular_value * right[column];
      }
    }
  }

  boundary->smoothing.resize(keys);
  std::vector<float> smoothed(residual.size());
  for (std::size_t row = 0; row < keys; ++row) {
    double mean_absolute = 0.0;
    for (std::size_t column = 0; column < values; ++column) {
      mean_absolute += std::abs(residual[row * values + column]);
    }
    mean_absolute /= static_cast<double>(values);
    const float smoothing = std::sqrt(std::max(
        static_cast<float>(mean_absolute), parameters.smoothing_floor));
    boundary->smoothing[row] = smoothing;
    for (std::size_t column = 0; column < values; ++column) {
      smoothed[row * values + column] =
          residual[row * values + column] / smoothing;
    }
  }

  boundary->value_scales.assign(values, 0.0F);
  boundary->residual.resize(keys * values);
  for (std::size_t column = 0; column < values; ++column) {
    float maximum = 0.0F;
    for (std::size_t row = 0; row < keys; ++row) {
      maximum = std::max(maximum,
                         std::abs(smoothed[row * values + column]));
    }
    boundary->value_scales[column] = maximum;
    for (std::size_t row = 0; row < keys; ++row) {
      const float scaled = maximum == 0.0F
                               ? 0.0F
                               : smoothed[row * values + column] /
                                     maximum * kQuantizedMaximum;
      const long quantized = std::lround(scaled);
      boundary->residual[row * values + column] =
          static_cast<std::int8_t>(std::clamp(quantized, -127L, 127L));
    }
  }
  return boundary;
}

std::vector<float> DequantizeBoundary(
    const LeapQuantCapsule::Boundary& boundary,
    const LeapQuantIdentity& identity,
    const LeapQuantParameters& parameters) {
  const std::size_t keys = identity.key_dimension;
  const std::size_t values = identity.value_dimension;
  std::vector<float> dense(keys * values);
  for (std::size_t row = 0; row < keys; ++row) {
    for (std::size_t column = 0; column < values; ++column) {
      float value = boundary.smoothing[row] *
                    static_cast<float>(boundary.residual[row * values + column]) /
                    kQuantizedMaximum * boundary.value_scales[column];
      for (std::size_t rank = 0; rank < parameters.compensator_rank; ++rank) {
        value += boundary.compensator_keys
                     [row * parameters.compensator_rank + rank] *
                 boundary.compensator_values
                     [column * parameters.compensator_rank + rank];
      }
      dense[row * values + column] = value;
    }
  }
  return dense;
}

// Evaluate S^T x directly from the fixed low-bit boundary and the buffered
// rank-one records. This is the paper's window readout: it never materializes
// the dense state between window boundaries.
std::vector<float> ReadWindowState(
    const LeapQuantCapsule::Boundary& boundary,
    std::span<const LeapQuantCapsule::UpdateRecord> updates,
    std::span<const float> read, const LeapQuantIdentity& identity,
    const LeapQuantParameters& parameters) {
  const std::size_t keys = identity.key_dimension;
  const std::size_t values = identity.value_dimension;
  std::vector<float> suffix_read(read.begin(), read.end());
  std::vector<float> output(values, 0.0F);

  for (std::size_t update_index = updates.size(); update_index > 0;
       --update_index) {
    const auto& update = updates[update_index - 1];
    float projection = 0.0F;
    for (std::size_t row = 0; row < keys; ++row) {
      projection += update.key[row] * suffix_read[row];
    }
    for (std::size_t column = 0; column < values; ++column) {
      output[column] += update.correction[column] * projection;
    }
    for (std::size_t row = 0; row < keys; ++row) {
      suffix_read[row] *= update.decay[row];
    }
  }

  for (std::size_t column = 0; column < values; ++column) {
    float residual_projection = 0.0F;
    for (std::size_t row = 0; row < keys; ++row) {
      residual_projection +=
          static_cast<float>(boundary.residual[row * values + column]) *
          boundary.smoothing[row] * suffix_read[row];
    }
    output[column] += residual_projection / kQuantizedMaximum *
                      boundary.value_scales[column];
  }
  for (std::size_t rank = 0; rank < parameters.compensator_rank; ++rank) {
    float projection = 0.0F;
    for (std::size_t row = 0; row < keys; ++row) {
      projection +=
          boundary.compensator_keys[row * parameters.compensator_rank + rank] *
          suffix_read[row];
    }
    for (std::size_t column = 0; column < values; ++column) {
      output[column] +=
          boundary.compensator_values
              [column * parameters.compensator_rank + rank] *
          projection;
    }
  }
  return output;
}

}  // namespace

LeapQuantCapsule::LeapQuantCapsule(
    std::shared_ptr<const Boundary> boundary, LeapQuantIdentity identity,
    LeapQuantParameters parameters, std::vector<UpdateRecord> updates)
    : boundary_(std::move(boundary)),
      identity_(std::move(identity)),
      parameters_(parameters),
      updates_(std::move(updates)) {}

LeapQuantCapsule LeapQuantCapsule::FromDenseBoundary(
    std::span<const float> dense_state, LeapQuantIdentity identity,
    LeapQuantParameters parameters) {
  ValidateParameters(parameters);
  ValidateIdentity(identity);
  Require(dense_state.size() ==
              static_cast<std::size_t>(identity.key_dimension) *
                  identity.value_dimension,
          "LeapQuant dense state geometry differs");
  Require(parameters.compensator_rank <=
              std::min(identity.key_dimension, identity.value_dimension),
          "LeapQuant compensator rank exceeds state geometry");
  for (const float value : dense_state) {
    Require(std::isfinite(value), "LeapQuant dense state is non-finite");
  }
  return LeapQuantCapsule(QuantizeBoundary(dense_state, identity, parameters),
                          std::move(identity), parameters, {});
}

std::vector<float> LeapQuantCapsule::DenseState() const {
  std::vector<float> state =
      DequantizeBoundary(*boundary_, identity_, parameters_);
  const std::size_t keys = identity_.key_dimension;
  const std::size_t values = identity_.value_dimension;
  for (const auto& update : updates_) {
    for (std::size_t row = 0; row < keys; ++row) {
      for (std::size_t column = 0; column < values; ++column) {
        state[row * values + column] =
            update.decay[row] * state[row * values + column] +
            update.key[row] * update.correction[column];
      }
    }
  }
  return state;
}

LeapQuantStepResult LeapQuantCapsule::ApplyToken(
    const LeapQuantTokenInput& input) {
  const std::size_t keys = identity_.key_dimension;
  const std::size_t values = identity_.value_dimension;
  Require(input.decay.size() == keys && input.key.size() == keys &&
              input.read.size() == keys && input.query.size() == keys &&
              input.value.size() == values,
          "LeapQuant token geometry differs");
  Require(AllFinite(input.decay) && AllFinite(input.key) &&
              AllFinite(input.value) && AllFinite(input.read) &&
              AllFinite(input.query),
          "LeapQuant token contains non-finite values");
  UpdateRecord update{input.decay, input.key, input.value};
  const auto state_read = ReadWindowState(*boundary_, updates_, input.read,
                                          identity_, parameters_);
  for (std::size_t column = 0; column < values; ++column) {
    update.correction[column] -= state_read[column];
  }
  LeapQuantStepResult result;
  std::vector<float> decayed_query(keys);
  float update_projection = 0.0F;
  for (std::size_t row = 0; row < keys; ++row) {
    decayed_query[row] = input.decay[row] * input.query[row];
    update_projection += input.key[row] * input.query[row];
  }
  result.output = ReadWindowState(*boundary_, updates_, decayed_query,
                                  identity_, parameters_);
  for (std::size_t column = 0; column < values; ++column) {
    result.output[column] += update.correction[column] * update_projection;
  }
  Require(AllFinite(result.output), "LeapQuant output is non-finite");
  if (updates_.size() + 1 == parameters_.window_size) {
    updates_.push_back(std::move(update));
    std::vector<float> state = DenseState();
    Require(AllFinite(state), "LeapQuant updated state is non-finite");
    auto next_boundary = QuantizeBoundary(state, identity_, parameters_);
    boundary_ = std::move(next_boundary);
    updates_.clear();
    result.crossed_window_boundary = true;
  } else {
    updates_.push_back(std::move(update));
  }
  return result;
}

LeapQuantCapsule LeapQuantCapsule::Fork() const { return *this; }

bool LeapQuantCapsule::SharesBoundaryWith(
    const LeapQuantCapsule& other) const noexcept {
  return boundary_.get() == other.boundary_.get();
}

const LeapQuantIdentity& LeapQuantCapsule::identity() const noexcept {
  return identity_;
}

const LeapQuantParameters& LeapQuantCapsule::parameters() const noexcept {
  return parameters_;
}

std::size_t LeapQuantCapsule::window_position() const noexcept {
  return updates_.size();
}

std::size_t LeapQuantCapsule::fp32_state_bytes() const noexcept {
  return static_cast<std::size_t>(identity_.key_dimension) *
         identity_.value_dimension * sizeof(float);
}

std::size_t LeapQuantCapsule::compressed_boundary_bytes() const noexcept {
  return boundary_->residual.size() * sizeof(std::int8_t) +
         (boundary_->smoothing.size() + boundary_->value_scales.size() +
          boundary_->compensator_keys.size() +
          boundary_->compensator_values.size()) *
             sizeof(float);
}

std::size_t LeapQuantCapsule::branch_private_bytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& update : updates_) {
    bytes += (update.decay.size() + update.key.size() +
              update.correction.size()) *
             sizeof(float);
  }
  return bytes;
}

void LeapQuantCapsule::RebindEpoch(std::uint64_t epoch) {
  Require(epoch > 0, "LeapQuant rebound epoch must be positive");
  identity_.epoch = epoch;
}

std::vector<std::uint8_t> LeapQuantCapsule::Serialize() const {
  std::vector<std::uint8_t> output(kMagic.begin(), kMagic.end());
  AppendInteger(output, parameters_.schema_version);
  AppendInteger(output, parameters_.window_size);
  AppendInteger(output, parameters_.compensator_rank);
  AppendInteger(output, parameters_.quant_bits);
  AppendFloat(output, parameters_.smoothing_floor);
  AppendInteger(output, identity_.generation);
  AppendInteger(output, identity_.epoch);
  AppendInteger(output, identity_.key_dimension);
  AppendInteger(output, identity_.value_dimension);
  AppendInteger(output, static_cast<std::uint32_t>(updates_.size()));
  output.insert(output.end(), identity_.model_geometry_sha256.begin(),
                identity_.model_geometry_sha256.end());
  for (const std::int8_t value : boundary_->residual) {
    output.push_back(static_cast<std::uint8_t>(value));
  }
  AppendFloats(output, boundary_->smoothing);
  AppendFloats(output, boundary_->value_scales);
  AppendFloats(output, boundary_->compensator_keys);
  AppendFloats(output, boundary_->compensator_values);
  for (const auto& update : updates_) {
    AppendFloats(output, update.decay);
    AppendFloats(output, update.key);
    AppendFloats(output, update.correction);
  }
  const std::string digest = BytesSha256(output);
  output.insert(output.end(), digest.begin(), digest.end());
  return output;
}

LeapQuantCapsule LeapQuantCapsule::Restore(
    std::span<const std::uint8_t> encoded,
    const LeapQuantIdentity& expected_identity,
    const LeapQuantParameters& expected_parameters,
    std::uint32_t expected_window_position) {
  ValidateIdentity(expected_identity);
  ValidateParameters(expected_parameters);
  Require(encoded.size() >= kMagic.size() + kDigestCharacters,
          "LeapQuant capsule is too short");
  Require(std::equal(kMagic.begin(), kMagic.end(), encoded.begin()),
          "LeapQuant capsule magic differs");
  const std::size_t content_size = encoded.size() - kDigestCharacters;
  const std::string recorded_digest(
      reinterpret_cast<const char*>(encoded.data() + content_size),
      kDigestCharacters);
  Require(IsLowercaseSha256Hex(recorded_digest),
          "LeapQuant capsule checksum encoding is invalid");
  Require(BytesSha256(encoded.first(content_size)) == recorded_digest,
          "LeapQuant capsule checksum differs");

  std::size_t offset = kMagic.size();
  LeapQuantParameters parameters;
  parameters.schema_version = ReadInteger<std::uint32_t>(encoded, offset);
  parameters.window_size = ReadInteger<std::uint32_t>(encoded, offset);
  parameters.compensator_rank = ReadInteger<std::uint32_t>(encoded, offset);
  parameters.quant_bits = ReadInteger<std::uint32_t>(encoded, offset);
  parameters.smoothing_floor = ReadFloat(encoded, offset);
  LeapQuantIdentity identity;
  identity.generation = ReadInteger<std::uint64_t>(encoded, offset);
  identity.epoch = ReadInteger<std::uint64_t>(encoded, offset);
  identity.key_dimension = ReadInteger<std::uint32_t>(encoded, offset);
  identity.value_dimension = ReadInteger<std::uint32_t>(encoded, offset);
  const auto update_count = ReadInteger<std::uint32_t>(encoded, offset);
  Require(offset <= content_size &&
              content_size - offset >= kDigestCharacters,
          "LeapQuant capsule geometry digest is truncated");
  identity.model_geometry_sha256.assign(
      reinterpret_cast<const char*>(encoded.data() + offset),
      kDigestCharacters);
  offset += kDigestCharacters;
  Require(parameters == expected_parameters,
          "LeapQuant capsule quantization parameters differ");
  Require(identity == expected_identity,
          "LeapQuant capsule identity or geometry differs");
  Require(parameters.compensator_rank <=
              std::min(identity.key_dimension, identity.value_dimension),
          "LeapQuant compensator rank exceeds restored geometry");
  Require(update_count < parameters.window_size,
          "LeapQuant capsule window position is invalid");
  Require(update_count == expected_window_position,
          "LeapQuant capsule window phase differs");

  const std::size_t keys = identity.key_dimension;
  const std::size_t values = identity.value_dimension;
  const std::size_t state_elements = keys * values;
  Require(offset <= content_size && content_size - offset >= state_elements,
          "LeapQuant capsule INT8 residual is truncated");
  auto boundary = std::make_shared<Boundary>();
  boundary->residual.resize(state_elements);
  for (std::int8_t& value : boundary->residual) {
    value = std::bit_cast<std::int8_t>(encoded[offset++]);
    Require(value != std::numeric_limits<std::int8_t>::min(),
            "LeapQuant INT8 residual contains -128");
  }
  boundary->smoothing = ReadFloats(encoded.first(content_size), offset, keys);
  boundary->value_scales =
      ReadFloats(encoded.first(content_size), offset, values);
  boundary->compensator_keys = ReadFloats(
      encoded.first(content_size), offset,
      keys * parameters.compensator_rank);
  boundary->compensator_values = ReadFloats(
      encoded.first(content_size), offset,
      values * parameters.compensator_rank);
  Require(AllFinite(boundary->smoothing) &&
              std::all_of(boundary->smoothing.begin(),
                          boundary->smoothing.end(),
                          [](float value) { return value > 0.0F; }) &&
              AllFinite(boundary->value_scales) &&
              std::all_of(boundary->value_scales.begin(),
                          boundary->value_scales.end(),
                          [](float value) { return value >= 0.0F; }) &&
              AllFinite(boundary->compensator_keys) &&
              AllFinite(boundary->compensator_values),
          "LeapQuant boundary payload is invalid");
  std::vector<UpdateRecord> updates;
  updates.reserve(update_count);
  for (std::uint32_t index = 0; index < update_count; ++index) {
    updates.push_back({ReadFloats(encoded.first(content_size), offset, keys),
                       ReadFloats(encoded.first(content_size), offset, keys),
                       ReadFloats(encoded.first(content_size), offset, values)});
    Require(AllFinite(updates.back().decay) && AllFinite(updates.back().key) &&
                AllFinite(updates.back().correction),
            "LeapQuant update record is non-finite");
  }
  Require(offset == content_size,
          "LeapQuant capsule contains trailing unverified payload");
  LeapQuantCapsule restored(std::move(boundary), std::move(identity),
                            parameters, std::move(updates));
  Require(AllFinite(restored.DenseState()),
          "LeapQuant restored state is non-finite");
  return restored;
}

LeapQuantBranch::LeapQuantBranch(LeapQuantCapsule capsule,
                                 std::uint64_t generation,
                                 std::uint64_t epoch)
    : capsule_(std::move(capsule)),
      expected_generation_(generation),
      expected_epoch_(epoch) {}

LeapQuantCapsule& LeapQuantBranch::capsule() noexcept { return capsule_; }

const LeapQuantCapsule& LeapQuantBranch::capsule() const noexcept {
  return capsule_;
}

bool LeapQuantBranch::active() const noexcept { return active_; }

LeapQuantCapsuleSlot::LeapQuantCapsuleSlot(LeapQuantCapsule capsule)
    : current_(std::move(capsule)) {}

LeapQuantBranch LeapQuantCapsuleSlot::BeginBranch() const {
  return LeapQuantBranch(current_.Fork(), current_.identity().generation,
                         current_.identity().epoch);
}

bool LeapQuantCapsuleSlot::Commit(LeapQuantBranch& branch) {
  if (!branch.active_ ||
      branch.expected_generation_ != current_.identity().generation ||
      branch.expected_epoch_ != current_.identity().epoch ||
      branch.capsule_.identity_ != current_.identity_ ||
      branch.capsule_.parameters_ != current_.parameters_ ||
      current_.identity().epoch == std::numeric_limits<std::uint64_t>::max()) {
    branch.active_ = false;
    return false;
  }
  // Complete all potentially-throwing work before the publication point.
  LeapQuantCapsule next = branch.capsule_;
  next.RebindEpoch(current_.identity().epoch + 1);
  static_assert(std::is_nothrow_move_assignable_v<LeapQuantCapsule>);
  current_ = std::move(next);
  branch.active_ = false;
  return true;
}

bool LeapQuantCapsuleSlot::Cancel(LeapQuantBranch& branch) noexcept {
  if (!branch.active_) return false;
  const bool current =
      branch.expected_generation_ == current_.identity().generation &&
      branch.expected_epoch_ == current_.identity().epoch;
  branch.active_ = false;
  return current;
}

bool LeapQuantCapsuleSlot::ReplaceGeneration(LeapQuantCapsule capsule) {
  const auto& incoming = capsule.identity();
  const auto& existing = current_.identity();
  if (incoming.generation <= existing.generation ||
      incoming.model_geometry_sha256 != existing.model_geometry_sha256 ||
      incoming.key_dimension != existing.key_dimension ||
      incoming.value_dimension != existing.value_dimension ||
      capsule.parameters() != current_.parameters()) {
    return false;
  }
  current_ = std::move(capsule);
  return true;
}

const LeapQuantCapsule& LeapQuantCapsuleSlot::current() const noexcept {
  return current_;
}

}  // namespace statecentric

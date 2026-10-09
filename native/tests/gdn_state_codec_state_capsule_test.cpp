#include "statecentric/gdn_state_codec_state_capsule.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "statecentric/sha256.h"

namespace {

using statecentric::GdnStateCodecCapsule;
using statecentric::GdnStateCodecCapsuleSlot;
using statecentric::GdnStateCodecIdentity;
using statecentric::GdnStateCodecParameters;
using statecentric::GdnStateCodecTokenInput;

constexpr std::size_t kKeys = 8;
constexpr std::size_t kValues = 6;

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

GdnStateCodecIdentity Identity(std::uint64_t generation = 7,
                           std::uint64_t epoch = 11) {
  return {std::string(64, 'a'), generation, epoch, kKeys, kValues};
}

GdnStateCodecParameters Parameters() { return {1, 16, 4, 8, 1.0e-8F}; }

std::vector<float> InitialState() {
  std::vector<float> state(kKeys * kValues);
  for (std::size_t index = 0; index < state.size(); ++index) {
    const float base = std::sin(static_cast<float>(index + 1) * 0.37F) * 0.4F;
    const float outlier = index % 17 == 0 ? 3.5F : 0.0F;
    state[index] = base + outlier;
  }
  return state;
}

GdnStateCodecTokenInput Token(std::size_t step, float branch_bias = 0.0F) {
  GdnStateCodecTokenInput input;
  input.decay.resize(kKeys);
  input.key.resize(kKeys);
  input.read.resize(kKeys);
  input.query.resize(kKeys);
  input.value.resize(kValues);
  for (std::size_t row = 0; row < kKeys; ++row) {
    input.decay[row] = 0.985F - 0.001F * static_cast<float>((step + row) % 5);
    input.key[row] =
        0.025F * std::sin(static_cast<float>((step + 1) * (row + 1)) * 0.13F) +
        branch_bias;
    input.read[row] =
        0.02F * std::cos(static_cast<float>((step + 2) * (row + 1)) * 0.11F);
    input.query[row] =
        0.03F * std::sin(static_cast<float>((step + 3) * (row + 1)) * 0.07F);
  }
  for (std::size_t column = 0; column < kValues; ++column) {
    input.value[column] =
        0.2F * std::cos(static_cast<float>((step + 1) * (column + 2)) * 0.09F) +
        branch_bias;
  }
  return input;
}

std::vector<float> ApplyFp32(std::vector<float>& state,
                             const GdnStateCodecTokenInput& input) {
  std::vector<float> correction = input.value;
  for (std::size_t column = 0; column < kValues; ++column) {
    for (std::size_t row = 0; row < kKeys; ++row) {
      correction[column] -= state[row * kValues + column] * input.read[row];
    }
  }
  for (std::size_t row = 0; row < kKeys; ++row) {
    for (std::size_t column = 0; column < kValues; ++column) {
      state[row * kValues + column] =
          input.decay[row] * state[row * kValues + column] +
          input.key[row] * correction[column];
    }
  }
  std::vector<float> output(kValues, 0.0F);
  for (std::size_t column = 0; column < kValues; ++column) {
    for (std::size_t row = 0; row < kKeys; ++row) {
      output[column] += state[row * kValues + column] * input.query[row];
    }
  }
  return output;
}

float RelativeRootMeanSquare(std::span<const float> actual,
                             std::span<const float> expected) {
  Check(actual.size() == expected.size(), "comparison geometry differs");
  double error = 0.0;
  double reference = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference =
        static_cast<double>(actual[index]) - expected[index];
    error += difference * difference;
    reference += static_cast<double>(expected[index]) * expected[index];
  }
  return static_cast<float>(std::sqrt(error / std::max(reference, 1.0e-30)));
}

std::string FloatDigest(std::span<const float> values) {
  std::string bytes;
  bytes.reserve(values.size() * sizeof(float));
  for (const float value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
      bytes.push_back(static_cast<char>(bits >> (byte * 8U)));
    }
  }
  return statecentric::Sha256Hex(bytes);
}

std::string CanonicalFloatDigest(std::span<const float> values) {
  std::string bytes;
  bytes.reserve(values.size() * sizeof(std::int64_t));
  for (const float value : values) {
    const auto fixed = static_cast<std::int64_t>(std::llround(value * 1.0e4F));
    const auto bits = std::bit_cast<std::uint64_t>(fixed);
    for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
      bytes.push_back(static_cast<char>(bits >> (byte * 8U)));
    }
  }
  return statecentric::Sha256Hex(bytes);
}

void WriteFloat(std::vector<std::uint8_t>& encoded, std::size_t offset,
                float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  Check(offset <= encoded.size() && encoded.size() - offset >= sizeof(bits),
        "float patch offset is invalid");
  for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
    encoded[offset + byte] = static_cast<std::uint8_t>(bits >> (byte * 8U));
  }
}

void RefreshChecksum(std::vector<std::uint8_t>& encoded) {
  constexpr std::size_t digest_characters = 64;
  Check(encoded.size() > digest_characters, "encoded capsule is too short");
  const std::size_t content_size = encoded.size() - digest_characters;
  const std::string digest = statecentric::Sha256Hex(std::string_view(
      reinterpret_cast<const char*>(encoded.data()), content_size));
  std::copy(digest.begin(), digest.end(), encoded.begin() + content_size);
}

template <typename Function>
void CheckRejects(Function&& function, const char* message) {
  bool rejected = false;
  try {
    std::forward<Function>(function)();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, message);
}

void CheckTrajectory() {
  auto fp32 = InitialState();
  auto capsule = GdnStateCodecCapsule::FromDenseBoundary(
      fp32, Identity(), Parameters());
  std::vector<float> fp32_outputs;
  std::vector<float> quantized_outputs;
  std::size_t boundaries = 0;
  for (std::size_t step = 0; step < 256; ++step) {
    const auto input = Token(step);
    const auto expected = ApplyFp32(fp32, input);
    const auto actual = capsule.ApplyToken(input);
    fp32_outputs.insert(fp32_outputs.end(), expected.begin(), expected.end());
    quantized_outputs.insert(quantized_outputs.end(), actual.output.begin(),
                             actual.output.end());
    boundaries += actual.crossed_window_boundary ? 1U : 0U;
  }
  Check(boundaries == 16, "window-boundary count differs");
  Check(capsule.window_position() == 0, "window did not close at p=16");
  Check(RelativeRootMeanSquare(capsule.DenseState(), fp32) < 0.025F,
        "long-trajectory state error exceeded reference gate");
  Check(RelativeRootMeanSquare(quantized_outputs, fp32_outputs) < 0.025F,
        "long-trajectory output error exceeded reference gate");
  Check(std::abs(quantized_outputs.front() - 0.0243581F) < 1.0e-6F &&
            std::abs(quantized_outputs.back() + 0.00297543F) < 1.0e-6F,
        "fixed trajectory numeric oracle differs");
  const std::string output_digest = FloatDigest(quantized_outputs);
  const std::string canonical_output_digest =
      CanonicalFloatDigest(quantized_outputs);
  Check(canonical_output_digest ==
            "6ab3da8358cd8c5ca5394bf10ad78015bc3f556804da1075ba91d49a5db30561",
        "fixed full-trajectory oracle differs");
  std::cout << "GDN_STATE_CODEC_REFERENCE={\"steps\":256,\"window_size\":16,"
               "\"rank\":4,\"state_relative_rms\":"
            << RelativeRootMeanSquare(capsule.DenseState(), fp32)
            << ",\"output_relative_rms\":"
            << RelativeRootMeanSquare(quantized_outputs, fp32_outputs)
            << ",\"first_output\":" << quantized_outputs.front()
            << ",\"last_output\":" << quantized_outputs.back()
            << ",\"output_sha256\":\"" << output_digest
            << "\",\"canonical_output_sha256\":\""
            << canonical_output_digest
            << "\"}\n";
}

void CheckForkLifecycle() {
  auto source = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), Identity(), Parameters());
  auto before = source.Fork();
  Check(source.SharesBoundaryWith(before),
        "fork-before did not share immutable boundary");
  Check(before.branch_private_bytes() == 0,
        "fork-before unexpectedly copied private records");

  for (std::size_t step = 0; step < 5; ++step) source.ApplyToken(Token(step));
  auto inside = source.Fork();
  Check(source.SharesBoundaryWith(inside),
        "fork-inside did not share immutable boundary");
  Check(inside.branch_private_bytes() == source.branch_private_bytes() &&
            inside.branch_private_bytes() > 0,
        "fork-inside did not copy only branch-local records");
  source.ApplyToken(Token(5, 0.01F));
  inside.ApplyToken(Token(5, -0.01F));
  Check(RelativeRootMeanSquare(source.DenseState(), inside.DenseState()) > 1.0e-5F,
        "forked branches did not diverge");

  for (std::size_t step = 6; step < 16; ++step) {
    source.ApplyToken(Token(step));
  }
  Check(source.window_position() == 0,
        "fork-at-boundary source did not close its window");
  auto at_boundary = source.Fork();
  Check(source.SharesBoundaryWith(at_boundary),
        "fork-at-boundary did not share the new boundary");

  GdnStateCodecCapsuleSlot slot(source);
  auto winner = slot.BeginBranch();
  auto late = slot.BeginBranch();
  winner.capsule().ApplyToken(Token(16, 0.02F));
  late.capsule().ApplyToken(Token(16, -0.02F));
  Check(slot.Commit(winner), "current branch commit was rejected");
  Check(slot.current().identity().epoch == Identity().epoch + 1,
        "successful commit did not advance epoch");
  Check(!slot.Commit(winner), "branch committed a second time");
  const auto committed = slot.current().Serialize();
  winner.capsule().ApplyToken(Token(17, 0.03F));
  Check(slot.current().Serialize() == committed,
        "inactive branch mutation changed committed state");
  Check(!slot.Commit(late), "late completion crossed epoch fence");
  Check(!late.active(), "late completion remained active");
  auto cancelled = slot.BeginBranch();
  Check(slot.Cancel(cancelled), "current branch cancel was rejected");
  Check(!slot.Commit(cancelled), "cancelled branch committed");

  auto foreign = slot.BeginBranch();
  auto foreign_identity = slot.current().identity();
  foreign_identity.model_geometry_sha256 = std::string(64, 'd');
  foreign.capsule() = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), foreign_identity, Parameters());
  Check(!slot.Commit(foreign), "foreign branch capsule committed");
  Check(slot.current().identity().model_geometry_sha256 == std::string(64, 'a'),
        "foreign branch capsule changed slot identity");

  auto stale_generation = slot.BeginBranch();
  auto replacement = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), Identity(8, 1), Parameters());
  Check(slot.ReplaceGeneration(std::move(replacement)),
        "newer slot generation was rejected");
  Check(!slot.Commit(stale_generation),
        "stale branch crossed the generation fence");

  auto nonadvancing = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), Identity(8, 2), Parameters());
  Check(!slot.ReplaceGeneration(std::move(nonadvancing)),
        "non-advancing generation replaced the slot");

  auto maximum_epoch = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), Identity(9, std::numeric_limits<std::uint64_t>::max()),
      Parameters());
  GdnStateCodecCapsuleSlot rollover_slot(std::move(maximum_epoch));
  auto rollover = rollover_slot.BeginBranch();
  Check(!rollover_slot.Commit(rollover), "epoch rollover was admitted");
  Check(!rollover.active(), "epoch-rollover branch remained active");
}

void CheckHibernateResume() {
  auto capsule = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), Identity(), Parameters());
  for (std::size_t step = 0; step < 7; ++step) capsule.ApplyToken(Token(step));
  const auto encoded = capsule.Serialize();
  const auto restored =
      GdnStateCodecCapsule::Restore(encoded, Identity(), Parameters(), 7);
  Check(restored.Serialize() == encoded,
        "hibernate/resume capsule did not round-trip exactly");
  Check(restored.window_position() == 7,
        "hibernate/resume window position differs");

  auto corrupt = encoded;
  corrupt[corrupt.size() / 2] ^= 0x01U;
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(corrupt, Identity(), Parameters(), 7);
      },
      "corrupt capsule was admitted");
  auto foreign_schema = encoded;
  foreign_schema[8] = 2;
  RefreshChecksum(foreign_schema);
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(foreign_schema, Identity(), Parameters(),
                                        7);
      },
      "foreign capsule schema was admitted");
  auto invalid_residual = encoded;
  constexpr std::size_t residual_offset = 120;
  invalid_residual[residual_offset] = 0x80U;
  RefreshChecksum(invalid_residual);
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(invalid_residual, Identity(),
                                        Parameters(), 7);
      },
      "out-of-domain INT8 residual was admitted");
  auto overflow = encoded;
  constexpr std::size_t smoothing_offset = residual_offset + kKeys * kValues;
  constexpr std::size_t value_scale_offset =
      smoothing_offset + kKeys * sizeof(float);
  overflow[residual_offset] = 0x7fU;
  WriteFloat(overflow, smoothing_offset, std::numeric_limits<float>::max());
  WriteFloat(overflow, value_scale_offset,
             std::numeric_limits<float>::max());
  RefreshChecksum(overflow);
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(overflow, Identity(), Parameters(), 7);
      },
      "checksum-valid overflowing payload was admitted");
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(encoded, Identity(8, 11), Parameters(),
                                        7);
      },
      "generation-mismatched capsule was admitted");
  auto wrong_parameters = Parameters();
  wrong_parameters.window_size = 8;
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(encoded, Identity(), wrong_parameters,
                                        7);
      },
      "parameter-mismatched capsule was admitted");
  auto wrong_geometry = Identity();
  wrong_geometry.model_geometry_sha256 = std::string(64, 'b');
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(encoded, wrong_geometry, Parameters(),
                                        7);
      },
      "model-mismatched capsule was admitted");
  CheckRejects(
      [&] {
        (void)GdnStateCodecCapsule::Restore(encoded, Identity(), Parameters(), 6);
      },
      "window-phase-mismatched capsule was admitted");
}

void CheckBoundaryFailureAtomicity() {
  auto capsule = GdnStateCodecCapsule::FromDenseBoundary(
      InitialState(), Identity(), Parameters());
  for (std::size_t step = 0; step < 15; ++step) capsule.ApplyToken(Token(step));
  const auto before = capsule.Serialize();
  auto overflow = Token(15);
  std::fill(overflow.read.begin(), overflow.read.end(), 0.0F);
  std::fill(overflow.key.begin(), overflow.key.end(),
            std::numeric_limits<float>::max());
  std::fill(overflow.value.begin(), overflow.value.end(),
            std::numeric_limits<float>::max());
  CheckRejects([&] { (void)capsule.ApplyToken(overflow); },
               "overflowing boundary token was admitted");
  Check(capsule.Serialize() == before,
        "failed boundary update partially mutated the capsule");
  Check(capsule.window_position() == 15,
        "failed boundary update changed the window position");
}

void CheckMemoryAccounting() {
  GdnStateCodecIdentity qwen_head{std::string(64, 'c'), 1, 1, 128, 128};
  std::vector<float> state(128 * 128, 0.0F);
  const auto capsule =
      GdnStateCodecCapsule::FromDenseBoundary(state, qwen_head, Parameters());
  Check(capsule.fp32_state_bytes() == 65'536,
        "Qwen head FP32 byte count differs");
  Check(capsule.compressed_boundary_bytes() == 21'504,
        "reference capsule boundary byte count differs");
  Check(capsule.compressed_boundary_bytes() < capsule.fp32_state_bytes(),
        "reference capsule did not compress boundary state");
}

}  // namespace

int main() {
  CheckTrajectory();
  CheckForkLifecycle();
  CheckHibernateResume();
  CheckBoundaryFailureAtomicity();
  CheckMemoryAccounting();
  return 0;
}

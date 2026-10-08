#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace statecentric {

struct LeapQuantParameters {
  std::uint32_t schema_version = 1;
  std::uint32_t window_size = 16;
  std::uint32_t compensator_rank = 4;
  std::uint32_t quant_bits = 8;
  float smoothing_floor = 1.0e-8F;

  bool operator==(const LeapQuantParameters&) const = default;
};

struct LeapQuantIdentity {
  std::string model_geometry_sha256;
  std::uint64_t generation = 0;
  std::uint64_t epoch = 0;
  std::uint32_t key_dimension = 0;
  std::uint32_t value_dimension = 0;

  bool operator==(const LeapQuantIdentity&) const = default;
};

struct LeapQuantTokenInput {
  std::vector<float> decay;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> read;
  std::vector<float> query;
};

struct LeapQuantStepResult {
  std::vector<float> output;
  bool crossed_window_boundary = false;
};

class LeapQuantCapsule {
 public:
  // Exposed as incomplete types so the CPU reference implementation can use
  // free numerical helpers without exposing their representation to callers.
  struct Boundary;
  struct UpdateRecord {
    std::vector<float> decay;
    std::vector<float> key;
    std::vector<float> correction;
  };

  static LeapQuantCapsule FromDenseBoundary(
      std::span<const float> dense_state, LeapQuantIdentity identity,
      LeapQuantParameters parameters = {});

  static LeapQuantCapsule Restore(
      std::span<const std::uint8_t> encoded,
      const LeapQuantIdentity& expected_identity,
      const LeapQuantParameters& expected_parameters,
      std::uint32_t expected_window_position);

  LeapQuantStepResult ApplyToken(const LeapQuantTokenInput& input);
  LeapQuantCapsule Fork() const;
  std::vector<float> DenseState() const;
  std::vector<std::uint8_t> Serialize() const;

  bool SharesBoundaryWith(const LeapQuantCapsule& other) const noexcept;
  const LeapQuantIdentity& identity() const noexcept;
  const LeapQuantParameters& parameters() const noexcept;
  std::size_t window_position() const noexcept;
  std::size_t fp32_state_bytes() const noexcept;
  std::size_t compressed_boundary_bytes() const noexcept;
  std::size_t branch_private_bytes() const noexcept;

 private:
  LeapQuantCapsule(std::shared_ptr<const Boundary> boundary,
                   LeapQuantIdentity identity,
                   LeapQuantParameters parameters,
                   std::vector<UpdateRecord> updates);
  void RebindEpoch(std::uint64_t epoch);

  std::shared_ptr<const Boundary> boundary_;
  LeapQuantIdentity identity_;
  LeapQuantParameters parameters_;
  std::vector<UpdateRecord> updates_;

  friend class LeapQuantCapsuleSlot;
};

class LeapQuantBranch {
 public:
  LeapQuantCapsule& capsule() noexcept;
  const LeapQuantCapsule& capsule() const noexcept;
  bool active() const noexcept;

 private:
  LeapQuantBranch(LeapQuantCapsule capsule, std::uint64_t generation,
                  std::uint64_t epoch);

  LeapQuantCapsule capsule_;
  std::uint64_t expected_generation_ = 0;
  std::uint64_t expected_epoch_ = 0;
  bool active_ = true;

  friend class LeapQuantCapsuleSlot;
};

class LeapQuantCapsuleSlot {
 public:
  explicit LeapQuantCapsuleSlot(LeapQuantCapsule capsule);

  LeapQuantBranch BeginBranch() const;
  bool Commit(LeapQuantBranch& branch);
  bool Cancel(LeapQuantBranch& branch) noexcept;
  bool ReplaceGeneration(LeapQuantCapsule capsule);
  const LeapQuantCapsule& current() const noexcept;

 private:
  LeapQuantCapsule current_;
};

}  // namespace statecentric

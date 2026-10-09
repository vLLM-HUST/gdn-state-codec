#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace statecentric {

struct GdnStateCodecParameters {
  std::uint32_t schema_version = 1;
  std::uint32_t window_size = 16;
  std::uint32_t compensator_rank = 4;
  std::uint32_t quant_bits = 8;
  float smoothing_floor = 1.0e-8F;

  bool operator==(const GdnStateCodecParameters&) const = default;
};

struct GdnStateCodecIdentity {
  std::string model_geometry_sha256;
  std::uint64_t generation = 0;
  std::uint64_t epoch = 0;
  std::uint32_t key_dimension = 0;
  std::uint32_t value_dimension = 0;

  bool operator==(const GdnStateCodecIdentity&) const = default;
};

struct GdnStateCodecTokenInput {
  std::vector<float> decay;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> read;
  std::vector<float> query;
};

struct GdnStateCodecStepResult {
  std::vector<float> output;
  bool crossed_window_boundary = false;
};

class GdnStateCodecCapsule {
 public:
  // Exposed as incomplete types so the CPU reference implementation can use
  // free numerical helpers without exposing their representation to callers.
  struct Boundary;
  struct UpdateRecord {
    std::vector<float> decay;
    std::vector<float> key;
    std::vector<float> correction;
  };

  static GdnStateCodecCapsule FromDenseBoundary(
      std::span<const float> dense_state, GdnStateCodecIdentity identity,
      GdnStateCodecParameters parameters = {});

  static GdnStateCodecCapsule Restore(
      std::span<const std::uint8_t> encoded,
      const GdnStateCodecIdentity& expected_identity,
      const GdnStateCodecParameters& expected_parameters,
      std::uint32_t expected_window_position);

  GdnStateCodecStepResult ApplyToken(const GdnStateCodecTokenInput& input);
  GdnStateCodecCapsule Fork() const;
  std::vector<float> DenseState() const;
  std::vector<std::uint8_t> Serialize() const;

  bool SharesBoundaryWith(const GdnStateCodecCapsule& other) const noexcept;
  const GdnStateCodecIdentity& identity() const noexcept;
  const GdnStateCodecParameters& parameters() const noexcept;
  std::size_t window_position() const noexcept;
  std::size_t fp32_state_bytes() const noexcept;
  std::size_t compressed_boundary_bytes() const noexcept;
  std::size_t branch_private_bytes() const noexcept;

 private:
  GdnStateCodecCapsule(std::shared_ptr<const Boundary> boundary,
                   GdnStateCodecIdentity identity,
                   GdnStateCodecParameters parameters,
                   std::vector<UpdateRecord> updates);
  void RebindEpoch(std::uint64_t epoch);

  std::shared_ptr<const Boundary> boundary_;
  GdnStateCodecIdentity identity_;
  GdnStateCodecParameters parameters_;
  std::vector<UpdateRecord> updates_;

  friend class GdnStateCodecCapsuleSlot;
};

class GdnStateCodecBranch {
 public:
  GdnStateCodecCapsule& capsule() noexcept;
  const GdnStateCodecCapsule& capsule() const noexcept;
  bool active() const noexcept;

 private:
  GdnStateCodecBranch(GdnStateCodecCapsule capsule, std::uint64_t generation,
                  std::uint64_t epoch);

  GdnStateCodecCapsule capsule_;
  std::uint64_t expected_generation_ = 0;
  std::uint64_t expected_epoch_ = 0;
  bool active_ = true;

  friend class GdnStateCodecCapsuleSlot;
};

class GdnStateCodecCapsuleSlot {
 public:
  explicit GdnStateCodecCapsuleSlot(GdnStateCodecCapsule capsule);

  GdnStateCodecBranch BeginBranch() const;
  bool Commit(GdnStateCodecBranch& branch);
  bool Cancel(GdnStateCodecBranch& branch) noexcept;
  bool ReplaceGeneration(GdnStateCodecCapsule capsule);
  const GdnStateCodecCapsule& current() const noexcept;

 private:
  GdnStateCodecCapsule current_;
};

}  // namespace statecentric

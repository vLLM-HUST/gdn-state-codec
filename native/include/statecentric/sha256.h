#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace statecentric {

class Sha256 {
 public:
  Sha256();

  void Update(std::span<const std::uint8_t> input);
  std::string FinishHex();

 private:
  void Transform(const std::uint8_t *block);

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> pending_{};
  std::size_t pending_bytes_ = 0;
  std::uint64_t total_bytes_ = 0;
  bool finished_ = false;
};

// Returns the SHA-256 digest of input as 64 lowercase hexadecimal digits.
std::string Sha256Hex(std::string_view input);

// Streams a regular file through SHA-256 without materializing it in memory.
std::string Sha256File(const std::filesystem::path &path);

bool IsLowercaseSha256Hex(std::string_view value) noexcept;

}  // namespace statecentric

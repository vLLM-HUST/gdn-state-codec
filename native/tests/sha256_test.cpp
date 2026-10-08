#include "statecentric/sha256.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  using statecentric::IsLowercaseSha256Hex;
  using statecentric::Sha256Hex;

  Check(Sha256Hex("") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "empty input vector");
  Check(Sha256Hex("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "abc vector");
  statecentric::Sha256 incremental;
  const std::string first = "a";
  const std::string second = "bc";
  incremental.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(first.data()), first.size()));
  incremental.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(second.data()), second.size()));
  Check(incremental.FinishHex() == Sha256Hex("abc"),
        "incremental vector");
  Check(Sha256Hex(
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "multi-block vector");
  Check(Sha256Hex(std::string(1'000'000, 'a')) ==
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "million-a vector");

  const std::string binary("a\0b", 3);
  Check(Sha256Hex(binary) ==
            "59b271ae1bbcb1d31d41929817f4b16fb439eb4f31520b5ad1d5ce98920a7138",
        "embedded NUL is part of the input");

  const auto digest = Sha256Hex("canonical");
  Check(IsLowercaseSha256Hex(digest), "generated digest is canonical");
  Check(!IsLowercaseSha256Hex(digest.substr(1)), "short digest is rejected");
  auto uppercase = digest;
  uppercase[0] = 'A';
  Check(!IsLowercaseSha256Hex(uppercase), "uppercase digest is rejected");
  auto punctuation = digest;
  punctuation[0] = ':';
  Check(!IsLowercaseSha256Hex(punctuation),
        "non-hexadecimal digest is rejected");

  const auto path = std::filesystem::temp_directory_path() /
                    "statecentric-sha256-test.bin";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "abc";
  }
  Check(statecentric::Sha256File(path) == Sha256Hex("abc"), "file vector");
  std::filesystem::remove(path);

  std::cout << "sha256 tests passed\n";
  return 0;
}

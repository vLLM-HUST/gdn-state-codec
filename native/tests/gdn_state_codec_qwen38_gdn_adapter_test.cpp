#include "statecentric/gdn_state_codec_qwen38_gdn_adapter.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void CheckNear(float actual, float expected, const char* message) {
  if (std::abs(actual - expected) > 1.0e-6F) {
    throw std::runtime_error(message);
  }
}

void CheckNonSquareLayoutRoundTrip() {
  constexpr std::size_t keys = 3;
  constexpr std::size_t values = 2;
  const std::vector<float> provider_value_key{1, 2, 3, 4, 5, 6};
  const auto gdn_state_codec = statecentric::Qwen38ProviderStateToGdnStateCodec(
      provider_value_key, keys, values);
  Check(gdn_state_codec == std::vector<float>({1, 4, 2, 5, 3, 6}),
        "provider [value,key] transpose differs");
  Check(statecentric::GdnStateCodecStateToQwen38Provider(gdn_state_codec, keys, values) ==
            provider_value_key,
        "Qwen GDN state layout round-trip differs");
}

void CheckProviderRecurrenceEquivalence() {
  constexpr std::size_t keys = 3;
  constexpr std::size_t values = 2;
  std::vector<float> provider{1, 2, 3, 4, 5, 6};
  const std::vector<float> query{0.1F, 0.5F, -0.2F};
  const std::vector<float> key{0.2F, -0.3F, 0.4F};
  const std::vector<float> value{0.7F, -0.2F};
  constexpr float decay = 0.9F;
  constexpr float beta = 0.6F;
  constexpr float scale = 0.5F;

  auto gdn_state_codec =
      statecentric::Qwen38ProviderStateToGdnStateCodec(provider, keys, values);
  const auto input = statecentric::Qwen38GdnTokenToGdnStateCodec(
      std::log(decay), beta, scale, query, key, value);

  std::vector<float> correction(values);
  for (std::size_t row = 0; row < values; ++row) {
    for (std::size_t column = 0; column < keys; ++column) {
      provider[row * keys + column] *= decay;
    }
    float projection = 0.0F;
    for (std::size_t column = 0; column < keys; ++column) {
      projection += provider[row * keys + column] * key[column];
    }
    correction[row] = beta * (value[row] - projection);
    for (std::size_t column = 0; column < keys; ++column) {
      provider[row * keys + column] += correction[row] * key[column];
    }
  }
  std::vector<float> provider_output(values);
  for (std::size_t row = 0; row < values; ++row) {
    for (std::size_t column = 0; column < keys; ++column) {
      provider_output[row] +=
          provider[row * keys + column] * query[column] * scale;
    }
  }

  for (std::size_t column = 0; column < values; ++column) {
    float projection = 0.0F;
    for (std::size_t row = 0; row < keys; ++row) {
      projection += gdn_state_codec[row * values + column] * input.read[row];
    }
    const float update = input.value[column] - projection;
    for (std::size_t row = 0; row < keys; ++row) {
      gdn_state_codec[row * values + column] =
          input.decay[row] * gdn_state_codec[row * values + column] +
          input.key[row] * update;
    }
  }
  std::vector<float> gdn_state_codec_output(values);
  for (std::size_t column = 0; column < values; ++column) {
    for (std::size_t row = 0; row < keys; ++row) {
      gdn_state_codec_output[column] +=
          gdn_state_codec[row * values + column] * input.query[row];
    }
  }

  const auto restored =
      statecentric::GdnStateCodecStateToQwen38Provider(gdn_state_codec, keys, values);
  for (std::size_t index = 0; index < provider.size(); ++index) {
    CheckNear(restored[index], provider[index],
              "Qwen GDN mapped state update differs");
  }
  for (std::size_t index = 0; index < values; ++index) {
    CheckNear(gdn_state_codec_output[index], provider_output[index],
              "Qwen GDN mapped output differs");
  }
}

}  // namespace

int main() {
  CheckNonSquareLayoutRoundTrip();
  CheckProviderRecurrenceEquivalence();
  return 0;
}

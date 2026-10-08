#include "statecentric/leapquant_qwen38_gdn_adapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace statecentric {
namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

bool AllFinite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

std::vector<float> Qwen38ProviderStateToLeapQuant(
    std::span<const float> provider_value_key, std::size_t key_dimension,
    std::size_t value_dimension) {
  Require(key_dimension > 0 && value_dimension > 0,
          "Qwen GDN state geometry must be nonzero");
  Require(provider_value_key.size() == key_dimension * value_dimension,
          "Qwen GDN provider state geometry differs");
  Require(AllFinite(provider_value_key),
          "Qwen GDN provider state is non-finite");
  std::vector<float> leapquant(key_dimension * value_dimension);
  for (std::size_t value = 0; value < value_dimension; ++value) {
    for (std::size_t key = 0; key < key_dimension; ++key) {
      leapquant[key * value_dimension + value] =
          provider_value_key[value * key_dimension + key];
    }
  }
  return leapquant;
}

std::vector<float> LeapQuantStateToQwen38Provider(
    std::span<const float> leapquant_key_value, std::size_t key_dimension,
    std::size_t value_dimension) {
  Require(key_dimension > 0 && value_dimension > 0,
          "LeapQuant state geometry must be nonzero");
  Require(leapquant_key_value.size() == key_dimension * value_dimension,
          "LeapQuant state geometry differs");
  Require(AllFinite(leapquant_key_value),
          "LeapQuant state is non-finite");
  std::vector<float> provider(key_dimension * value_dimension);
  for (std::size_t key = 0; key < key_dimension; ++key) {
    for (std::size_t value = 0; value < value_dimension; ++value) {
      provider[value * key_dimension + key] =
          leapquant_key_value[key * value_dimension + value];
    }
  }
  return provider;
}

LeapQuantTokenInput Qwen38GdnTokenToLeapQuant(
    float log_decay, float beta, float query_scale,
    std::span<const float> normalized_query,
    std::span<const float> normalized_key,
    std::span<const float> value) {
  Require(!normalized_key.empty() && !value.empty(),
          "Qwen GDN token geometry must be nonzero");
  Require(normalized_query.size() == normalized_key.size(),
          "Qwen GDN query/key geometry differs");
  Require(std::isfinite(log_decay) && std::isfinite(beta) &&
              std::isfinite(query_scale) && AllFinite(normalized_query) &&
              AllFinite(normalized_key) && AllFinite(value),
          "Qwen GDN token is non-finite");
  const float decay = std::exp(log_decay);
  Require(std::isfinite(decay), "Qwen GDN decay is non-finite");

  LeapQuantTokenInput input;
  input.decay.assign(normalized_key.size(), decay);
  input.key.resize(normalized_key.size());
  input.read.resize(normalized_key.size());
  input.query.resize(normalized_query.size());
  input.value.assign(value.begin(), value.end());
  for (std::size_t index = 0; index < normalized_key.size(); ++index) {
    input.key[index] = beta * normalized_key[index];
    input.read[index] = decay * normalized_key[index];
    input.query[index] = query_scale * normalized_query[index];
  }
  Require(AllFinite(input.key) && AllFinite(input.read) &&
              AllFinite(input.query),
          "Qwen GDN mapped token is non-finite");
  return input;
}

}  // namespace statecentric

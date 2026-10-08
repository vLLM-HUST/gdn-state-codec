#pragma once

#include <span>
#include <vector>

#include "statecentric/leapquant_state_capsule.h"

namespace statecentric {

// Qwen recurrent state is provider-major [value,key]. LeapQuant stores the
// same matrix transposed as [key,value].
std::vector<float> Qwen38ProviderStateToLeapQuant(
    std::span<const float> provider_value_key, std::size_t key_dimension,
    std::size_t value_dimension);

std::vector<float> LeapQuantStateToQwen38Provider(
    std::span<const float> leapquant_key_value, std::size_t key_dimension,
    std::size_t value_dimension);

// The provider applies decay before reading the recurrent state:
//   R <- exp(g) R
//   delta <- beta * (v - R k)
//   R <- R + delta k^T
//   o <- scale * R q
// For S=R^T, the equivalent LeapQuant record is decay=exp(g),
// write_key=beta*k, read=exp(g)*k, value=v, query=scale*q.
LeapQuantTokenInput Qwen38GdnTokenToLeapQuant(
    float log_decay, float beta, float query_scale,
    std::span<const float> normalized_query,
    std::span<const float> normalized_key,
    std::span<const float> value);

}  // namespace statecentric

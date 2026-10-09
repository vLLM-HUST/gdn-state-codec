#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "statecentric/gdn_state_codec_windowed_gdn_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr std::uint32_t kDimension = 128;
constexpr std::uint32_t kColumns = 16;
constexpr std::uint32_t kScalarStride = 8;

struct Runtime {
  std::uint32_t states = 0;
  std::uint32_t references = 0;
  std::int8_t* quantized = nullptr;
  std::int32_t* projected = nullptr;
  float* dynamic_scale = nullptr;
  void* workspace = nullptr;
  void* tiling = nullptr;
  std::size_t workspace_bytes = 0;
  std::size_t tiling_bytes = 0;
};

std::mutex runtime_mutex;
Runtime* shared_runtime = nullptr;

void Release(Runtime* runtime) {
  if (runtime == nullptr) return;
  if (runtime->tiling != nullptr) (void)aclrtFree(runtime->tiling);
  if (runtime->workspace != nullptr) (void)aclrtFree(runtime->workspace);
  if (runtime->dynamic_scale != nullptr)
    (void)aclrtFree(runtime->dynamic_scale);
  if (runtime->projected != nullptr) (void)aclrtFree(runtime->projected);
  if (runtime->quantized != nullptr) (void)aclrtFree(runtime->quantized);
  delete runtime;
}

bool Allocate(void** pointer, std::size_t bytes) {
  return aclrtMalloc(pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

std::vector<std::uint8_t> MakeTiling(
    const platform_ascendc::PlatformAscendC& platform) {
  optiling::TCubeTiling tiling;
  matmul_tiling::MultiCoreMatmulTiling generator(platform);
  generator.SetDim(static_cast<int32_t>(platform.GetCoreNumAiv()));
  generator.SetAType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT8, false);
  generator.SetBType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT8, false);
  generator.SetCType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_INT32);
  generator.SetOrgShape(kDimension, kColumns, kDimension);
  generator.SetShape(kDimension, kColumns, kDimension);
  generator.SetBias(false);
  generator.SetBufferSpace(-1, -1, -1);
  if (generator.GetTiling(tiling) < 0) return {};
  std::vector<std::uint8_t> output(tiling.GetDataSize());
  tiling.SaveToBuffer(output.data(), output.size());
  return output;
}
}  // namespace

extern "C" void* statecentric_gdn_state_codec_decode_runtime_create_v1(
    std::uint32_t states) {
  if (states == 0 || states > 16384) return nullptr;
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (shared_runtime != nullptr) {
    if (shared_runtime->states != states) return nullptr;
    ++shared_runtime->references;
    return shared_runtime;
  }
  auto* platform =
      platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B2");
  if (platform == nullptr) return nullptr;
  auto tiling = MakeTiling(*platform);
  if (tiling.empty()) return nullptr;
  auto* runtime = new Runtime;
  runtime->states = states;
  runtime->references = 1;
  runtime->workspace_bytes = platform->GetLibApiWorkSpaceSize();
  runtime->tiling_bytes = tiling.size();
  const std::size_t matrix_elements =
      static_cast<std::size_t>(states) * kDimension * kColumns;
  if (!Allocate(reinterpret_cast<void**>(&runtime->quantized),
                matrix_elements * sizeof(std::int8_t)) ||
      !Allocate(reinterpret_cast<void**>(&runtime->projected),
                matrix_elements * sizeof(std::int32_t)) ||
      !Allocate(reinterpret_cast<void**>(&runtime->dynamic_scale),
                static_cast<std::size_t>(states) * kScalarStride *
                    sizeof(float)) ||
      !Allocate(&runtime->workspace, runtime->workspace_bytes) ||
      !Allocate(&runtime->tiling, runtime->tiling_bytes) ||
      aclrtMemcpy(runtime->tiling, runtime->tiling_bytes, tiling.data(),
                  tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    Release(runtime);
    return nullptr;
  }
  shared_runtime = runtime;
  return runtime;
}

extern "C" void statecentric_gdn_state_codec_decode_runtime_destroy_v1(
    void* opaque) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  auto* runtime = static_cast<Runtime*>(opaque);
  if (runtime == nullptr || runtime != shared_runtime) return;
  if (--runtime->references != 0) return;
  shared_runtime = nullptr;
  Release(runtime);
}

extern "C" std::int32_t statecentric_gdn_state_codec_decode_runtime_step_v1(
    void* opaque, void* stream, const std::int8_t* residual,
    const float* smoothing, const float* scales,
    const std::uint16_t* compensator_keys,
    const std::uint16_t* compensator_values, float* record_decay,
    std::uint16_t* record_keys, std::uint16_t* record_corrections,
    std::uint32_t* positions, const float* decay,
    const std::uint16_t* beta, const std::uint16_t* keys,
    const std::uint16_t* queries, const std::uint16_t* values,
    std::uint16_t* outputs,
    std::uint32_t states) {
  auto* runtime = static_cast<Runtime*>(opaque);
  if (runtime == nullptr || runtime != shared_runtime || !stream ||
      states != runtime->states) return 1;
  if (statecentric_gdn_state_codec_prepare_decode_pair_bf16_launch_v1(
          stream, keys, queries, decay, smoothing, runtime->quantized,
          runtime->dynamic_scale, states) != 0) return 2;
  if (statecentric_gdn_state_codec_int8_cube_matmul_launch_v1(
          stream, residual, runtime->quantized, runtime->projected, states, 0,
          runtime->workspace, runtime->tiling, (states + 1U) / 2U) != 0)
    return 3;
  if (statecentric_gdn_state_codec_finish_decode_pair_bf16_launch_v1(
          stream, runtime->projected, runtime->dynamic_scale, scales,
          compensator_keys, compensator_values, record_decay, record_keys,
          record_corrections, positions, decay, beta, keys, queries,
          values, outputs, states) != 0) return 4;
  return 0;
}

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
constexpr std::uint32_t kWindow = 16;

struct Runtime {
  std::uint32_t states = 0;
  std::uint32_t references = 0;
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
                     matmul_tiling::DataType::DT_FLOAT16, true);
  generator.SetBType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16, false);
  generator.SetCType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
  generator.SetOrgShape(kDimension, kDimension, kWindow);
  generator.SetShape(kDimension, kDimension, kWindow);
  generator.SetBias(false);
  generator.SetBufferSpace(-1, -1, -1);
  if (generator.GetTiling(tiling) < 0) return {};
  std::vector<std::uint8_t> output(tiling.GetDataSize());
  tiling.SaveToBuffer(output.data(), output.size());
  return output;
}
}  // namespace

extern "C" void* statecentric_gdn_state_codec_boundary_runtime_create_v1(
    std::uint32_t states) {
  if (states == 0 || states > 128) return nullptr;
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
  if (!Allocate(&runtime->workspace, runtime->workspace_bytes) ||
      !Allocate(&runtime->tiling, runtime->tiling_bytes) ||
      aclrtMemcpy(runtime->tiling, runtime->tiling_bytes, tiling.data(),
                  tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    Release(runtime);
    return nullptr;
  }
  shared_runtime = runtime;
  return runtime;
}

extern "C" void statecentric_gdn_state_codec_boundary_runtime_destroy_v1(
    void* opaque) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  auto* runtime = static_cast<Runtime*>(opaque);
  if (runtime == nullptr || runtime != shared_runtime) return;
  if (--runtime->references != 0) return;
  shared_runtime = nullptr;
  Release(runtime);
}

extern "C" std::int32_t statecentric_gdn_state_codec_boundary_runtime_step_v1(
    void* opaque, void* stream, const float* record_decay,
    const std::uint16_t* record_keys,
    const std::uint16_t* record_corrections,
    std::uint16_t* scaled_record_keys, float* record_merge,
    std::uint32_t states) {
  auto* runtime = static_cast<Runtime*>(opaque);
  if (runtime == nullptr || runtime != shared_runtime || !stream ||
      !record_decay || !record_keys || !record_corrections ||
      !scaled_record_keys || !record_merge ||
      states != runtime->states) {
    return 1;
  }
  if (statecentric_gdn_state_codec_boundary_scale_records_launch_v1(
          stream, record_decay, record_keys, scaled_record_keys, states) != 0) {
    return 2;
  }
  if (statecentric_gdn_state_codec_bf16_cube_matmul_launch_v1(
          stream, record_corrections, scaled_record_keys, record_merge,
          states, runtime->workspace, runtime->tiling,
          (states + 1U) / 2U) != 0) {
    return 3;
  }
  return 0;
}

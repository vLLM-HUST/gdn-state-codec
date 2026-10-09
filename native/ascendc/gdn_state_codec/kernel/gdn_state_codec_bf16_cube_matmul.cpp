#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

namespace {
__aicore__ inline void CopyTiling(TCubeTiling* destination, GM_ADDR source) {
  auto* output = reinterpret_cast<uint32_t*>(destination);
  auto* input = reinterpret_cast<__gm__ uint32_t*>(source);
  for (uint32_t index = 0; index < sizeof(TCubeTiling) / sizeof(uint32_t);
       ++index) {
    output[index] = input[index];
  }
}
}  // namespace

// Batched FP16 record merge. A is stored [K,M], B is [K,N], and C is FP32
// [M,N]. The Cube path transposes A without an explicit packing kernel.
extern "C" __global__ __aicore__ void
statecentric_gdn_state_codec_bf16_cube_matmul(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t states,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  AscendC::TPipe pipe;
  TCubeTiling tiling;
  CopyTiling(&tiling, tiling_gm);

  AscendC::GlobalTensor<half> a_global;
  AscendC::GlobalTensor<half> b_global;
  AscendC::GlobalTensor<float> c_global;
  a_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a),
                           states * tiling.M * tiling.Ka);
  b_global.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b),
                           states * tiling.Kb * tiling.N);
  c_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c),
                           states * tiling.M * tiling.N);

  const uint32_t state = AscendC::GetBlockIdx();
  const uint32_t offset_a = state * tiling.M * tiling.Ka;
  const uint32_t offset_b = state * tiling.Kb * tiling.N;
  const uint32_t offset_c = state * tiling.M * tiling.N;

  Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>> matmul;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmul, &tiling);
  if (state >= states) return;
  matmul.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
  matmul.SetTensorA(a_global[offset_a], true);
  matmul.SetTensorB(b_global[offset_b], false);
  matmul.SetTail(tiling.M, tiling.N);
  matmul.IterateAll(c_global[offset_c]);
  matmul.End();
}

extern "C" int32_t statecentric_gdn_state_codec_bf16_cube_matmul_launch_v1(
    void* stream, const uint16_t* a, const uint16_t* b, float* c,
    uint32_t states, void* workspace, void* tiling, uint32_t blocks) {
  if (!stream || !a || !b || !c || !workspace || !tiling || blocks == 0 ||
      blocks > 64 || states == 0 || states > 128 ||
      blocks != (states + 1U) / 2U) {
    return 1;
  }
  statecentric_gdn_state_codec_bf16_cube_matmul<<<blocks, nullptr, stream>>>(
      reinterpret_cast<half*>(const_cast<uint16_t*>(a)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(b)), c, states,
      workspace, tiling);
  return 0;
}

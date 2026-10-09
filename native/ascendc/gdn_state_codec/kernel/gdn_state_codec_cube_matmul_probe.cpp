#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

namespace {
__aicore__ inline uint32_t Ceiling(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

__aicore__ inline void CopyTiling(TCubeTiling* destination,
                                  GM_ADDR source) {
  auto* output = reinterpret_cast<uint32_t*>(destination);
  auto* input = reinterpret_cast<__gm__ uint32_t*>(source);
  for (uint32_t index = 0; index < sizeof(TCubeTiling) / sizeof(uint32_t);
       ++index) {
    output[index] = input[index];
  }
}
}  // namespace

// Correctness probe for the Cube ABI used by the GDN State Codec boundary refit.
// A is [M,K] FP16, B is [K,N] FP16, and C is [M,N] FP32 in ND format.
extern "C" __global__ __aicore__ void statecentric_gdn_state_codec_cube_matmul_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t states, uint32_t transpose_a,
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
  const uint32_t remaining_m = tiling.M;
  const uint32_t remaining_n = tiling.N;
  const uint32_t tail_m = remaining_m < tiling.singleCoreM
                              ? remaining_m
                              : tiling.singleCoreM;
  const uint32_t tail_n = remaining_n < tiling.singleCoreN
                              ? remaining_n
                              : tiling.singleCoreN;

  Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>> matmul;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmul, &tiling);
  // Registration must happen on both vector sub-blocks before the unused
  // one returns; otherwise an odd usedCoreNum leaves the Cube KFC server
  // waiting forever.
  if (state >= states) return;
  matmul.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
  matmul.SetTensorA(a_global[offset_a], transpose_a != 0U);
  matmul.SetTensorB(b_global[offset_b], false);
  matmul.SetTail(tail_m, tail_n);
  matmul.IterateAll(c_global[offset_c]);
  matmul.End();
}

extern "C" int32_t statecentric_gdn_state_codec_cube_matmul_probe_launch_v1(
    void* stream, const uint16_t* a, const uint16_t* b, float* c,
    uint32_t states, uint32_t transpose_a, void* workspace, void* tiling,
    uint32_t blocks) {
  if (!stream || !a || !b || !c || !workspace || !tiling || blocks == 0 ||
      blocks > 64 || states == 0 || states > 128 ||
      blocks != (states + 1U) / 2U || transpose_a > 1U) {
    return 1;
  }
  // The generated launcher recognizes the explicit `workspace` argument and
  // installs its system prefix. The middle launch-control slot is not a
  // workspace pointer and must remain null.
  statecentric_gdn_state_codec_cube_matmul_probe<<<blocks, nullptr, stream>>>(
      reinterpret_cast<half*>(const_cast<uint16_t*>(a)),
      reinterpret_cast<half*>(const_cast<uint16_t*>(b)), c, states,
      transpose_a, workspace, tiling);
  return 0;
}

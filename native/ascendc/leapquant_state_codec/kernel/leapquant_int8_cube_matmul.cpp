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

// Batched INT8 residual projection. A is [M,K], B is padded [K,16], and C is
// [M,16] INT32. Each logical state occupies one Cube task.
extern "C" __global__ __aicore__ void
statecentric_leapquant_int8_cube_matmul(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t states, uint32_t transpose_a,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  AscendC::TPipe pipe;
  TCubeTiling tiling;
  CopyTiling(&tiling, tiling_gm);

  AscendC::GlobalTensor<int8_t> a_global;
  AscendC::GlobalTensor<int8_t> b_global;
  AscendC::GlobalTensor<int32_t> c_global;
  a_global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),
                           states * tiling.M * tiling.Ka);
  b_global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),
                           states * tiling.Kb * tiling.N);
  c_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(c),
                           states * tiling.M * tiling.N);

  const uint32_t state = AscendC::GetBlockIdx();
  const uint32_t offset_a = state * tiling.M * tiling.Ka;
  const uint32_t offset_b = state * tiling.Kb * tiling.N;
  const uint32_t offset_c = state * tiling.M * tiling.N;

  Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>> matmul;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmul, &tiling);
  if (state >= states) return;
  matmul.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
  matmul.SetTensorA(a_global[offset_a], transpose_a != 0U);
  matmul.SetTensorB(b_global[offset_b], false);
  matmul.SetTail(tiling.M, tiling.N);
  matmul.IterateAll(c_global[offset_c]);
  matmul.End();
}

extern "C" int32_t statecentric_leapquant_int8_cube_matmul_launch_v1(
    void* stream, const int8_t* a, const int8_t* b, int32_t* c,
    uint32_t states, uint32_t transpose_a, void* workspace, void* tiling,
    uint32_t blocks) {
  if (!stream || !a || !b || !c || !workspace || !tiling || blocks == 0 ||
      blocks > 64 || states == 0 || states > 128 ||
      blocks != (states + 1U) / 2U || transpose_a > 1U) {
    return 1;
  }
  statecentric_leapquant_int8_cube_matmul<<<blocks, nullptr, stream>>>(
      const_cast<int8_t*>(a), const_cast<int8_t*>(b), c, states, transpose_a,
      workspace, tiling);
  return 0;
}

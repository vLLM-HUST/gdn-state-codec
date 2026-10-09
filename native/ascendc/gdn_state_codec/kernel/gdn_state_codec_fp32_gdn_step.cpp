#include "kernel_operator.h"
#include "statecentric/gdn_state_codec_windowed_gdn_kernel.h"

namespace { constexpr uint32_t kD = 128; constexpr uint32_t kE = kD * kD;
class Fp32Step {
 public:
  __aicore__ inline void Init(GM_ADDR state, GM_ADDR decay, GM_ADDR key,
      GM_ADDR read, GM_ADDR query, GM_ADDR value, GM_ADDR output, uint32_t states) {
    states_=states; state_.SetGlobalBuffer((__gm__ float*)state,states*kE);
    decay_.SetGlobalBuffer((__gm__ float*)decay,states); key_.SetGlobalBuffer((__gm__ half*)key,states*kD);
    read_.SetGlobalBuffer((__gm__ half*)read,states*kD); query_.SetGlobalBuffer((__gm__ half*)query,states*kD);
    value_.SetGlobalBuffer((__gm__ half*)value,states*kD); output_.SetGlobalBuffer((__gm__ float*)output,states*kD);
  }
  __aicore__ inline void Process(){for(uint32_t s=AscendC::GetBlockIdx();s<states_;s+=AscendC::GetBlockNum()){
    uint32_t vb=s*kD,sb=s*kE; for(uint32_t c=0;c<kD;++c){float projection=0;
      for(uint32_t r=0;r<kD;++r) projection+=state_.GetValue(sb+r*kD+c)*static_cast<float>(read_.GetValue(vb+r));
      float correction=static_cast<float>(value_.GetValue(vb+c))-projection; float out=0;
      for(uint32_t r=0;r<kD;++r){float next=decay_.GetValue(s)*state_.GetValue(sb+r*kD+c)+static_cast<float>(key_.GetValue(vb+r))*correction;
        state_.SetValue(sb+r*kD+c,next); out+=next*static_cast<float>(query_.GetValue(vb+r));}
      output_.SetValue(vb+c,out);}}}
 private:uint32_t states_=0; AscendC::GlobalTensor<float> state_,decay_,output_; AscendC::GlobalTensor<half> key_,read_,query_,value_;};}
extern "C" __global__ __aicore__ void statecentric_gdn_state_codec_fp32_gdn_step(GM_ADDR state,GM_ADDR decay,GM_ADDR key,GM_ADDR read,GM_ADDR query,GM_ADDR value,GM_ADDR output,uint32_t states){Fp32Step k;k.Init(state,decay,key,read,query,value,output,states);k.Process();}
extern "C" int32_t statecentric_gdn_state_codec_fp32_gdn_step_launch_v1(void* stream,float* state,const float* decay,const uint16_t* key,const uint16_t* read,const uint16_t* query,const uint16_t* value,float* output,uint32_t states){if(!stream||!state||!decay||!key||!read||!query||!value||!output||states==0||states>16384)return 1;statecentric_gdn_state_codec_fp32_gdn_step<<<64,nullptr,stream>>>(state,const_cast<float*>(decay),reinterpret_cast<half*>(const_cast<uint16_t*>(key)),reinterpret_cast<half*>(const_cast<uint16_t*>(read)),reinterpret_cast<half*>(const_cast<uint16_t*>(query)),reinterpret_cast<half*>(const_cast<uint16_t*>(value)),output,states);return 0;}

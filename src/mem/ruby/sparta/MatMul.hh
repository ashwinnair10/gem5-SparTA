#ifndef __MATMUL_HH__
#define __MATMUL_HH__

#include "base/types.hh"
#include "mem/ruby/sparta/PE_Acc.hh"
#include "mem/ruby/sparta/PE_Mul.hh"
#include "params/MatMul.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class MatMul : public SimObject
  {
    public:
      PE_Mul *mul;
      PE_Acc *acc;
      float **A, **B, **C;
      int M, N, K;
      int i, j, k;
      bool done=false;
      std::function<void()> finishedCallback;

      void setFinishedCallback(std::function<void()> cb) {
          finishedCallback = cb;
      }


      MatMul(const MatMulParams &p);

      void startup() override;

      void startMatMul(uint64_t A_ptr,
                         uint64_t B_ptr,
                         uint64_t C_ptr,
                         int _M, int _N, int _K);

      void onMulDone(float product);

      void onAccDone(float partial,int remaining_ops);

      bool isDone() const { return done;}
  };
}

#endif

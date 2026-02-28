#ifndef __MATMUL_HH__
#define __MATMUL_HH__

#include <queue>

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/MatMul.hh"
#include "sim/sim_object.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

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
      bool transpose;
      std::function<void()> finishedCallback;
      std::queue<std::tuple<float,float,int>> stallMulQueue;
      std::queue<std::pair<float,int>> stallAccQueue;
      std::vector<float> partialSums;
      std::vector<int> remainingCounts;

      EventFunctionWrapper retryEvent;
      EventFunctionWrapper tickEvent;
      statistics::Scalar stallCycles;

      void setFinishedCallback(std::function<void()> cb) {
          finishedCallback = cb;
      }


      MatMul(const MatMulParams &p);

      void startup() override;

      void startMatMul(uint64_t A_ptr,
                         uint64_t B_ptr,
                         uint64_t C_ptr,
                         int _M, int _N, int _K,bool t);

      void onMulDone(float product,int idx);

      void onAccDone(float partial,int idx);

      bool isDone() const { return done;}

      void retryStalled();

    private:
        void regStats() override;
        void tick();
  };
}

#endif

#ifndef __PE_ACC_HH__
#define __PE_ACC_HH__

#include <queue>

#include "base/types.hh"
#include "params/PE_Acc.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class PE_Acc : public SimObject
  {
    public:

      std::function<void(float,int,int)> callback;
      std::queue<std::pair<float,int>> inputQueue;
      void setCallback(std::function<void(float,int,int)> cb)
      {
        callback = cb;
      }

      PE_Acc(const PE_AccParams &p);

      void startup() override;

      void processNext();

      void setParams(float sum, int remaining);

      void feedProduct(float product,int id);

      void reset(int num_ops);

      float getFinalResult() const {return current_sum;}

    private:
      Tick latency;
      int island;
      EventFunctionWrapper computeEvent;

      float current_sum;
      float current_input;
      int id;
      int remaining_ops;
      void finishCompute();
  };
}

#endif

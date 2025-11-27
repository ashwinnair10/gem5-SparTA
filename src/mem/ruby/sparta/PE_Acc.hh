#ifndef __PE_ACC_HH__
#define __PE_ACC_HH__

#include "base/types.hh"
#include "params/PE_Acc.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class PE_Acc : public SimObject
  {
    public:

      std::function<void(float,int)> callback;
      void setCallback(std::function<void(float,int)> cb) { callback = cb; }

      PE_Acc(const PE_AccParams &p);

      void startup() override;

      void feedProduct(float product);

      void reset(int num_ops);

      float getFinalResult() const {return current_sum;}

    private:
      Tick latency;
      EventFunctionWrapper computeEvent;

      float current_sum;
      float current_input;
      int remaining_ops;

      void finishCompute();
  };
}

#endif

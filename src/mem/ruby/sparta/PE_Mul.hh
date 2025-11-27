#ifndef __PE_MUL_HH__
#define __PE_MUL_HH__

#include "base/types.hh"
#include "params/PE_Mul.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class PE_Mul : public SimObject
  {
    public:

      std::function<void(float)> callback;
      void setCallback(std::function<void(float)> cb) { callback = cb; }

      PE_Mul(const PE_MulParams &p);

      void startup() override;

      void startCompute(float val1,float val2);

      float getResult() const {return result;}

    private:
      Tick latency;
      EventFunctionWrapper computeEvent;
      float operand1;
      float operand2;
      float result;

      void finishCompute();
  };
}

#endif

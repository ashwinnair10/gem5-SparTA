#ifndef __PE_MUL_HH__
#define __PE_MUL_HH__

#include <queue>

#include "base/types.hh"
#include "params/PE_Mul.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class PE_Mul : public SimObject
  {
    public:

      std::function<void(float,int)> callback;
      std::queue<std::tuple<float,float,int>> inputQueue;
      void setCallback(std::function<void(float,int)> cb) { callback = cb; }

      PE_Mul(const PE_MulParams &p);

      void startup() override;

      void startCompute(float val1,float val2,int id);

      float getResult() const {return result;}

    private:
      Tick latency;
      int island;
      EventFunctionWrapper computeEvent;
      float operand1;
      float operand2;
      int id;
      float result;
      void processNext();
      void finishCompute();
  };
}

#endif

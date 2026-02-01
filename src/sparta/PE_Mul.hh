#ifndef __PE_MUL_HH__
#define __PE_MUL_HH__

#include <queue>

#include "base/statistics.hh"
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


      bool push(float val1,float val2,int id);

      float getResult() const {return result;}

      bool isFull() const {
          return inputQueue.size() == queue_size;
      }

      void regStats() override;

    private:
      Tick latency;
      int island;
      int queue_size;
      EventFunctionWrapper computeEvent;
      EventFunctionWrapper tickEvent;
      bool busy;
      float operand1;
      float operand2;
      int id;
      float result;
      static constexpr const char* GREEN = "\033[32m";
      static constexpr const char* RESET = "\033[0m";
      void processNext();
      void tick();
      void finishCompute();

      statistics::Scalar numMulOps;
      statistics::Scalar activeCycles;
      statistics::Scalar idleCycles;


  };
}

#endif

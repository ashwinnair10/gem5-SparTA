#ifndef __PE_ACC_HH__
#define __PE_ACC_HH__

#include <queue>

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/PE_Acc.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class PE_Acc : public SimObject
  {
    public:

      std::function<void(float,int)> callback;
      std::queue<std::pair<float,int>> inputQueue;
      void setCallback(std::function<void(float,int)> cb)
      {
        callback = cb;
      }

      PE_Acc(const PE_AccParams &p);

      void startup() override;

      void processNext();

      bool push(float product,int id);

      float getFinalResult() const {return current_sum;}
      void regStats() override;
      bool isFull() const {
          return inputQueue.size() == queue_size;
      }

    private:
      Tick latency;
      int island;
      int queue_size;
      EventFunctionWrapper computeEvent;
      EventFunctionWrapper tickEvent;
      static constexpr const char* YELLOW = "\033[33m";
      static constexpr const char* RESET = "\033[0m";
      float current_sum;
      float current_input;
      int id;
      int remaining_ops;
      bool busy;
      void finishCompute();
      void tick();

      statistics::Scalar numAccOps;
      statistics::Scalar activeCycles;
      statistics::Scalar idleCycles;


  };
}

#endif

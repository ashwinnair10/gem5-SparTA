#ifndef __PE_MUL_HH__
#define __PE_MUL_HH__

#include "base/types.hh"
#include "params/PE_Mul.hh"
#include "sim/sim_object.hh"

namespace gem5{
  class PE_Mul : public SimObject
  {
    public:
      PE_Mul(const PE_MulParams &p);

      void startup() override;

      void startCompute(int block_id);

    private:
      Tick latency;
      EventFunctionWrapper computeEvent;

      void finishCompute();
  };
}

#endif

#ifndef __BASELINE_DRIVER_HH__
#define __BASELINE_DRIVER_HH__

#include "params/BaselineDriver.hh"
#include "sim/sim_object.hh"
#include "base/types.hh"
#include "mem/ruby/sparta/MatMul.hh"

namespace gem5 {

class BaselineDriver : public SimObject
{
  private:
    MatMul *mm;

    float **Q, **K, **V;
    float **Scores;
    float **Prob;
    float **Output;

    int M, N, Kdim;

    EventFunctionWrapper startEvent;

    enum Phase {
        PHASE_IDLE,
        PHASE_QK,
        PHASE_SOFTMAX,
        PHASE_AV,
        PHASE_DONE
    } phase;

    float *softmaxRow;

  public:
    BaselineDriver(const BaselineDriverParams &p);

    void startup() override;

    void start();

    void onQKDone();
    void onAVDone();

    void runSoftmax();
    void startAV();
};

}

#endif

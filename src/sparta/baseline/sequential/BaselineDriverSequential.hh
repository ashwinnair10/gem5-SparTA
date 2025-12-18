#ifndef __BASELINE_DRIVER_SEQUENTIAL_HH__
#define __BASELINE_DRIVER_SEQUENTIAL_HH__

#include "base/types.hh"
#include "params/BaselineDriverSequential.hh"
#include "sim/sim_object.hh"
#include "sparta/MatMul.hh"

namespace gem5 {

class BaselineDriverSequential : public SimObject
{
  private:
    MatMul *mm;

    float **Q, **K, **V;
    float **Scores;
    float **Prob;
    float **Output;

    int M, N, Kdim;

    EventFunctionWrapper startEvent;

    enum Phase
    {
        PHASE_IDLE,
        PHASE_QK,
        PHASE_SOFTMAX,
        PHASE_AV,
        PHASE_DONE
    } phase;

    float *softmaxRow;

  public:
    BaselineDriverSequential(const BaselineDriverSequentialParams &p);

    void startup() override;

    void start();

    void onQKDone();
    void onAVDone();

    void runSoftmax();
    void startAV();
};

}

#endif

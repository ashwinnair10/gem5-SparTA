#ifndef __BASELINE_DRIVER_SEQUENTIAL_HH__
#define __BASELINE_DRIVER_SEQUENTIAL_HH__

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/BaselineDriverSequential.hh"
#include "sim/sim_object.hh"
#include "sparta/MatMul.hh"

namespace gem5 {

class BaselineDriverSequential : public SimObject
{
  private:
    MatMul *mm;
    float **X;
    float **WQ;
    float **WK;
    float **WV;

    int projPart;   // 0=Q,1=K,2=V

    float **Q, **K, **V;
    float **Scores;
    float **Prob;
    float **Output;

    int M, N, Kdim;

    EventFunctionWrapper startEvent;
    EventFunctionWrapper tickEvent;

    enum Phase
    {
        PHASE_IDLE,
        PHASE_PROJ,
        PHASE_QK,
        PHASE_SOFTMAX,
        PHASE_AV,
        PHASE_DONE
    } phase;

    float *softmaxRow;

    statistics::Scalar numReads;
    statistics::Scalar numWrites;
    statistics::Scalar stallCycles;

    void regStats() override;

  public:
    BaselineDriverSequential(const BaselineDriverSequentialParams &p);

    void startup() override;

    void start();

    void tick();

    void startProjection();

    void onProjectionDone();

    void onQKDone();
    void onAVDone();

    void runSoftmax();
    void startAV();
};

}

#endif

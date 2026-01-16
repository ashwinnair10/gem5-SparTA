#ifndef __BASELINE_DRIVER_PARALLEL_HH__
#define __BASELINE_DRIVER_PARALLEL_HH__

#include <queue>

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/BaselineDriverParallel.hh"
#include "sim/sim_object.hh"

namespace gem5{
    class BaselineDriverParallel : public SimObject
    {
    public:

        std::vector<PE_Mul*> mulUnits;
        std::vector<PE_Acc*> accUnits;

        float **Q, **K, **V;
        float **Scores, **Prob, **Output;
        float **currentOut;

        int M, N, Kdim;
        int numPEs;
        int currentCols;

        uint64_t totalTasks;
        uint64_t completedTasks;

        std::queue<std::tuple<float,float,int>> stallMulQueue;
        std::queue<std::pair<float,int>> stallAccQueue;

        EventFunctionWrapper startEvent;
        EventFunctionWrapper retryEvent;

        enum Phase
        {
            PHASE_IDLE,
            PHASE_QK,
            PHASE_SOFTMAX,
            PHASE_AV,
            PHASE_DONE
        } phase;

        BaselineDriverParallel(const BaselineDriverParallelParams &p);

        void startup() override;

        void start();
        void startAV();
        void
        dispatchMatMul(float **A, float **B, float **C, int M, int N, int K);

        void onProductReady(int pe, float product,int idx);
        void onAccReady(int pe, float sum, int remaining,int idx);
        void retryStalled();

        void onQKDone();
        void onAVDone();
        void runSoftmax();

        float *softmaxRow;

    private:
        statistics::Scalar numReads;
        statistics::Scalar numWrites;
        statistics::Scalar stallCycles;

        void regStats() override;
    };
}

#endif

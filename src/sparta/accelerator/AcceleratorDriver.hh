#ifndef __ACCELERATOR_DRIVER_HH__
#define __ACCELERATOR_DRIVER_HH__

#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/AcceleratorDriver.hh"
#include "sim/sim_object.hh"

namespace gem5{
    class AcceleratorDriver : public SimObject
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

        EventFunctionWrapper startEvent;

        bool retryPending;

        struct MulTask
        {
            float a,b;
            int idx;
        };
        std::queue<MulTask> mulTaskQueue;

        struct AccTask
        {
            float product;
            int idx;
        };
        std::queue<AccTask> accTaskQueue;
        std::unordered_map<int, float> partialSums;
        std::unordered_map<int, int> remainingCounts;

        std::vector<int> accBusy;
        uint64_t hashSeed = 0;
        std::vector<int> accLoad;
        std::unordered_map<int, int> idxToAccPE;


        enum Phase
        {
            PHASE_IDLE,
            PHASE_QK,
            PHASE_SOFTMAX,
            PHASE_AV,
            PHASE_DONE
        } phase;

        AcceleratorDriver(const AcceleratorDriverParams &p);

        void startup() override;

        void start();
        void startAV();
        void
        dispatchMatMul(float **A, float **B, float **C, int M, int N, int K);

        void tryScheduleMul();
        void tryScheduleAcc();
        int findFreeAccPE(int idx);

        void onProductReady(int pe, float product,int idx);
        void onAccReady(int pe, float sum, int remaining,int idx);

        void onQKDone();
        void onAVDone();
        void runSoftmax();

        float *softmaxRow;

        static constexpr const char* RED = "\033[31m";
        static constexpr const char* RESET = "\033[0m";

        int hashToPE(int idx) const {
            return (idx ^ hashSeed) % numPEs;
        }

        void reseed();

    private:
        statistics::Scalar numReads;
        statistics::Scalar numWrites;
        statistics::Scalar stallCycles;

        void regStats() override;

    };
}

#endif

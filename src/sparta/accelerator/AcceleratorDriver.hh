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

        EventFunctionWrapper startEvent;
        EventFunctionWrapper tickEvent;

        enum Phase
        {
            PHASE_IDLE,
            PHASE_QK,
            PHASE_SOFTMAX,
            PHASE_AV,
            PHASE_DONE
        } phase;

        int currentCols;

        uint64_t totalTasks;
        uint64_t completedTasks;

        struct MulTask
        {
            float a,b;
            int idx;
        };

        std::queue<MulTask> mulTaskQueue;

        struct AccTask
        {
            float product;
            int sid;
        };
        std::queue<AccTask> accTaskQueue;

        std::vector<int> remaining;
        std::unordered_map<int,int> idxToSID;
        std::vector<int> accBusy;
        uint64_t hashSeed = 1632093731;
        std::vector<int> mulLoad;
        std::unordered_map<int, int> sidToAccPE;

        int maxLiveOps;
        std::queue<int> freeSIDs;
        std::vector<int> sidToIdx;
        std::vector<int> sidRemainingAcc;
        std::vector<int> sidRemainingMul;
        std::vector<float> sidPartialSum;


        AcceleratorDriver(const AcceleratorDriverParams &p);

        void startup() override;

        void start();
        void startAV();
        void
        dispatchMatMul(float **A, float **B, float **C, int M, int N, int K);

        void tryScheduleMul();
        void tryScheduleAcc();
        int findFreeAccPE(int sid);

        void onProductReady(int pe, float product,int sid);
        void onAccReady(int pe, float sum,int sid);

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

        int allocSID(int idx,int nnz);
        void freeSID(int sid);

        void tick();

    };
}

#endif

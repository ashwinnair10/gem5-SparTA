#ifndef __PADE_DRIVER_HH__
#define __PADE_DRIVER_HH__

#include <queue>

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/PadeDriver.hh"
#include "sim/sim_object.hh"

namespace gem5
{
class PadeDriver : public SimObject
{
  public:
    std::vector<PE_Mul *> mulUnits;
    std::vector<PE_Acc *> accUnits;

    float **X;
    float **WQ;
    float **WK;
    float **WV;

    int projPart;

    float **Q, **K, **V;
    float **Scores, **Prob, **Output;
    float **currentOut;
    float **currentA, **currentB;

    int M, N, Kdim;
    int currentK;
    int numPEs;
    int currentCols;
    bool isTranspose;

    uint64_t totalTasks;
    uint64_t completedTasks;

    std::queue<std::tuple<float, float, int>> stallMulQueue;
    std::queue<std::pair<float, int>> stallAccQueue;

    std::vector<float> partialSums;
    std::vector<int> remainingCounts;

    EventFunctionWrapper startEvent;
    EventFunctionWrapper retryEvent;
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

    struct PadeBitState
    {
        int current_bit_round = 0;
        int current_k;
        float partial_sum = 0.0f;
        float total_dot_prod = 0.0f;
        bool is_pruned = false;

        PadeBitState(int bit = 0, int k = 0, float psum = 0.0f,
                     float total = 0.0f, bool prune = false)
            : current_bit_round(bit),
              current_k(k),
              partial_sum(psum),
              total_dot_prod(total),
              is_pruned(prune)
        {}
    };

    std::map<int, PadeBitState> padeTracker;
    // std::queue<int> waitingTasks;
    std::vector<int> activeTasks;

    std::vector<float> rowMax;

    PadeDriver(const PadeDriverParams &p);

    void startup() override;

    void start();
    void startProjection();
    void onProjectionDone();
    void startAV();
    void dispatchMatMul(float **A, float **B, float **C, int M, int N, int K,
                        bool transpose = false);
    void issueBit(int idx, int pe);
    void tryDispatchNext();
    bool checkBUI_GF(int idx);
    void handleTaskCompletion(int idx);
    int pickBestTask();
    int findLeastLoadedPE();

    void onProductReady(int pe, float product, int idx);
    void onAccReady(int pe, float sum, int idx);
    void retryStalled();

    void onQKDone();
    void onAVDone();
    void runSoftmax();

    float *softmaxRow;

  private:
    statistics::Scalar numReads;
    statistics::Scalar numWrites;
    statistics::Scalar stallCycles;
    statistics::Scalar prunedTokens;

    const int PADE_GUARD_BIT = 3;
    const int TOTAL_BITS = 8;

    void regStats() override;
    void tick();
};
} // namespace gem5

#endif

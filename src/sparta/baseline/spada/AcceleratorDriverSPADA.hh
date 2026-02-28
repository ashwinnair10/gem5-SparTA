#ifndef __ACCELERATOR_DRIVER_SPADA_HH__
#define __ACCELERATOR_DRIVER_SPADA_HH__

#include <queue>
#include <unordered_map>
#include <vector>

#include "base/statistics.hh"
#include "params/AcceleratorDriverSPADA.hh"
#include "sim/sim_object.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5 {

class AcceleratorDriverSPADA : public SimObject
{
  public:
    AcceleratorDriverSPADA(const AcceleratorDriverSPADAParams &p);
    void startup() override;

  private:
    /* ---------------- Hardware ---------------- */
    std::vector<PE_Mul*> mulUnits;
    std::vector<PE_Acc*> accUnits;


    /* ---------------- Tensors ---------------- */
    float **Q, **K, **V;
    float **Scores, **Prob, **Output;
    float **currentOut;

    int M, N, Kdim;
    int numPEs;
    int currentCols;

    /* ---------------- Events ---------------- */
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

    /* ---------------- Sparse metadata (oracle) ---------------- */
    struct SparseRow
    {
        std::vector<int> cols;
    };
    std::vector<SparseRow> A_csr;
    std::vector<SparseRow> B_csr;

    /* ---------------- Tasks ---------------- */
    struct MulTask
    {
        float a, b;
        int idx;
    };
    struct AccTask
    {
        float val;
        int sid;
    };

    std::queue<MulTask> mulQ;
    std::queue<AccTask> accQ;

    std::unordered_map<int,int> idxToSID;
    std::vector<int> remaining;

    /* ---------------- SID bookkeeping ---------------- */
    int maxLiveOps;
    std::queue<int> freeSIDs;
    std::vector<int> sidToIdx;
    std::vector<int> sidRemaining;
    std::vector<float> sidPartial;

    /* ---------------- Progress ---------------- */
    uint64_t totalTasks = 0;
    uint64_t completedTasks = 0;

    /* ---------------- Stats ---------------- */
    statistics::Scalar stallCycles;
    statistics::Scalar numReads;
    statistics::Scalar numWrites;

    /* ---------------- Core logic ---------------- */
    void start();
    void tick();

    void buildSparseMeta(float **A, float **B, int M, int N, int K);
    void dispatchMatMulSPADA(float **A, float **B, float **C,
                             int M, int N, int K);

    void tryScheduleMul();
    void tryScheduleAcc();

    void onProductReady(int pe, float product, int sid);
    void onAccReady(int pe, float sum, int sid);

    void onQKDone();
    void onAVDone();
    void runSoftmax();
    void startAV();

    int allocSID(int idx, int nnz);
    void freeSID(int sid);

    void regStats() override;
};

} // namespace gem5

#endif

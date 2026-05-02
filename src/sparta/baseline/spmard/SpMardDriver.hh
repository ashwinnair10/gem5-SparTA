#ifndef __SPMARD_DRIVER_HH__
#define __SPMARD_DRIVER_HH__

#include <queue>

#include "base/statistics.hh"
#include "base/types.hh"
#include "params/SpMardDriver.hh"
#include "sim/sim_object.hh"

namespace gem5
{
class SpMardDriver : public SimObject
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

    int M, N, Kdim;
    int numPEs;
    int currentCols;

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

    enum Dataflow
    {
        IP_M,
        IP_N,
        OP_M,
        OP_N,
        ROW_M,
        ROW_N
    };

    struct CSR
    {
        std::vector<float> values;
        std::vector<int> col_idx;
        std::vector<int> row_ptr;
    };

    CSR A_csr;
    CSR B_csr;

    Dataflow currentDataflow;

    SpMardDriver(const SpMardDriverParams &p);

    void startup() override;

    void start();
    void startProjection();
    void onProjectionDone();
    void startAV();
    void dispatchMatMul(float **A, float **B, float **C, int M, int N, int K,
                        bool transpose = false);
    void dispatchIP(float **A, float **B, int M, int N, int K,
                    bool transpose = false, bool A_stationary = false);
    void dispatchOP(float **A, float **B, int M, int N, int K,
                    bool transpose = false, bool A_stationary = false);
    void dispatchROW(float **A, float **B, int M, int N, int K,
                     bool transpose = false, bool A_stationary = false);

    void onProductReady(int pe, float product, int idx);
    void onAccReady(int pe, float sum, int idx);
    void retryStalled();

    void onQKDone();
    void onAVDone();
    void runSoftmax();
    Dataflow chooseDataflow(float **A, float **B, int M, int N, int K,
                            bool transpose);
    void buildCSR(float **mat, int rows, int cols, CSR &csr);
    void buildCSRTranspose(float **mat, int rows, int cols, CSR &csr);
    int countNNZ(float **mat, int rows, int cols);

    float *softmaxRow;

  private:
    statistics::Scalar numReads;
    statistics::Scalar numWrites;
    statistics::Scalar stallCycles;

    void regStats() override;
    void tick();
};
} // namespace gem5

#endif

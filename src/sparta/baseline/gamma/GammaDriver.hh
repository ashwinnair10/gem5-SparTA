#ifndef __SPARTA_BASELINE_GAMMA_DRIVER_HH__
#define __SPARTA_BASELINE_GAMMA_DRIVER_HH__

#include <queue>
#include <unordered_map>
#include <vector>

#include "base/statistics.hh"
#include "params/GammaDriver.hh"
#include "sim/sim_object.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5 {

class GammaDriver : public SimObject
{
  public:

    GammaDriver(const GammaDriverParams &p);

    void startup() override;
    void regStats() override;

  private:

    //////////////////////////////////////////////////////
    // MATRIX POINTERS (same as baseline)

    float **X;
    float **WQ;
    float **WK;
    float **WV;

    int projPart;

    float **Q;
    float **K;
    float **V;

    float **Scores;
    float **Prob;
    float **Output;

    //////////////////////////////////////////////////////
    // CURRENT MATMUL STATE

    float **currentA;
    float **currentB;
    float **currentOut;

    int currentCols;
    int currentKDim;

    //////////////////////////////////////////////////////
    // DIMENSIONS

    int M;
    int N;
    int Kdim;

    int numPEs;

    //////////////////////////////////////////////////////
    // PHASE CONTROL

    enum Phase
    {
        PHASE_IDLE,
        PHASE_PROJ,
        PHASE_QK,
        PHASE_SOFTMAX,
        PHASE_AV,
        PHASE_DONE
    };

    Phase phase;

    bool transpose;



    //////////////////////////////////////////////////////
    // PE UNITS

    std::vector<PE_Mul*> mulUnits;
    std::vector<PE_Acc*> accUnits;

    //////////////////////////////////////////////////////
    // GAMMA STATE

    // which row each PE owns
    std::vector<int> rowAssigned;

    // remaining ops per PE
    std::vector<int> remainingOps;

    // local accumulator per PE
    std::vector<std::unordered_map<int,float>> localAcc;

    // row scheduler
    std::queue<int> rowQueue;

    std::vector<int> pe_k;
    std::vector<int> pe_j;
    std::vector<bool> issueDone;

    struct CSR
    {
        std::vector<float> values;
        std::vector<int> col_idx;
        std::vector<int> row_ptr;
    };

    CSR A_csr;
    CSR B_csr;

    static constexpr const char* RED = "\033[31m";
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";


    //////////////////////////////////////////////////////
    // STALL STRUCTURES

    struct MulStallTask
    {
        int pe;
        int row;
        int k;
        int j;
    };

    struct AccStallTask
    {
        int pe;
        float product;
        int col;
    };

    std::queue<MulStallTask> stallMulQueue;
    std::queue<AccStallTask> stallAccQueue;

    //////////////////////////////////////////////////////
    // EVENTS

    EventFunctionWrapper startEvent;
    EventFunctionWrapper retryEvent;
    EventFunctionWrapper tickEvent;

    //////////////////////////////////////////////////////
    // STATS

    statistics::Scalar stallCycles;

    //////////////////////////////////////////////////////
    // SOFTMAX BUFFER

    float *softmaxRow;

    //////////////////////////////////////////////////////
    // CORE FUNCTIONS

    void start();

    void tick();

    void startProjection();
    void onProjectionDone();

    void startAV();
    void onAVDone();

    void onQKDone();

    void runSoftmax();

    //////////////////////////////////////////////////////
    // GAMMA CORE FUNCTIONS

    void dispatchMatMul(int rows);

    void assignRows();

    void issueRow(int pe, int row);

    void finishRow(int pe);

    void retryStalled();

    void checkComplete();

    void buildCSR(float **mat, int rows, int cols, CSR &csr);
    void buildCSRTranspose(float **mat, int rows, int cols, CSR &csr);

    //////////////////////////////////////////////////////
    // CALLBACKS

    void onProductReady(int pe, float product, int col);

    void onAccReady(int pe, float sum, int col);
};

} // namespace gem5

#endif

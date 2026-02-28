#ifndef __BLOCK_SPADA_DRIVER_HH__
#define __BLOCK_SPADA_DRIVER_HH__

#include <queue>
#include <vector>

#include "base/statistics.hh"
#include "params/BlockSPADADriver.hh"
#include "sim/sim_object.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5 {

class BlockSPADADriver : public SimObject
{
  public:
    BlockSPADADriver(const BlockSPADADriverParams &p);
    void startup() override;

  private:
    /* ===== CONFIG ===== */
    static constexpr int BS = 16;

    /* ===== HARDWARE ===== */
    std::vector<PE_Mul*> mulUnits;
    std::vector<PE_Acc*> accUnits;


    /* ===== TENSORS ===== */
    float **Q, **K, **V;
    float **Scores, **Prob, **Output;
    float **curA, **curB, **curOut;

    int M, N, Kdim;
    int numPEs;
    int curCols;
    int innerDim;
    bool transposeB;

    /* ===== EVENTS ===== */
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

    /* ===== BLOCK STATE ===== */
    int bi, bj;      // block coordinates
    int ii, jj;      // element inside block
    int k;           // reduction index

    bool blockActive;
    bool elemActive;

    /* ===== SID ===== */
    int sid;
    float accValue;
    int remainingMul;

    /* ===== STATS ===== */
    statistics::Scalar stallCycles;

    /* ===== CORE ===== */
    void start();
    void tick();

    void startQK();
    void startAV();
    void runSoftmax();

    void advanceBlock();
    void advanceElement();
    void issueMul();

    void onProductReady(int pe, float product, int sid);
    void onAccReady(int pe, float sum, int sid);

    void regStats() override;
};

} // namespace gem5

#endif

#include "sparta/baseline/block-spada/BlockSPADADriver.hh"

#include <cassert>
#include <cmath>

#include "sim/sim_exit.hh"

namespace gem5 {

BlockSPADADriver::
BlockSPADADriver(const BlockSPADADriverParams &p)
  : SimObject(p),
    Q(reinterpret_cast<float **>(p.Q)),
    K(reinterpret_cast<float **>(p.K)),
    V(reinterpret_cast<float **>(p.V)),
    Scores(reinterpret_cast<float **>(p.Scores)),
    Prob(reinterpret_cast<float **>(p.Prob)),
    Output(reinterpret_cast<float **>(p.Output)),
    M(p.M), N(p.N), Kdim(p.Kdim),
    numPEs(p.numPEs),
    startEvent([this]{ start(); }, "blockspada_start"),
    tickEvent([this]{ tick(); }, "blockspada_tick"),
    phase(PHASE_IDLE),
    blockActive(false),
    elemActive(false)
{
    for (auto *m : p.mul_units) mulUnits.push_back(m);
    for (auto *a : p.acc_units) accUnits.push_back(a);
}

void BlockSPADADriver::startup()
{
    for (int i = 0; i < numPEs; i++) {
        mulUnits[i]->setCallback(
            [this,i](float p,int sid){ onProductReady(i,p,sid); });
        accUnits[i]->setCallback(
            [this,i](float s,int sid){ onAccReady(i,s,sid); });
    }

    schedule(startEvent, curTick()+1);
    schedule(tickEvent, curTick()+1);
}

void BlockSPADADriver::start()
{
    startQK();
}

void BlockSPADADriver::startQK()
{
    phase = PHASE_QK;
    curA = Q;
    curB = K;
    curOut = Scores;
    curCols = N;
    transposeB = false;
    innerDim = Kdim;

    bi = bj = ii = jj = k = 0;
    blockActive = true;
    elemActive = false;
}

void BlockSPADADriver::startAV()
{
    phase = PHASE_AV;
    curA = Prob;
    curB = V;
    curOut = Output;
    curCols = Kdim;
    transposeB = false;
    innerDim = N;

    bi = bj = ii = jj = k = 0;
    blockActive = true;
    elemActive = false;
}

void BlockSPADADriver::tick()
{
    if (phase == PHASE_DONE)
        return;

    if (!blockActive) {
        schedule(tickEvent, curTick()+1);
        return;
    }

    if (!elemActive) {
        advanceElement();
    } else {
        issueMul();
    }

    schedule(tickEvent, curTick()+1);
}

void BlockSPADADriver::advanceBlock()
{
    bj++;
    if (bj * BS >= curCols) {
        bj = 0;
        bi++;
    }

    if (bi * BS >= M) {
        blockActive = false;
        if (phase == PHASE_QK) {
            runSoftmax();
            startAV();
        } else {
            phase = PHASE_DONE;
            exitSimLoop("Block-SPADA done");
        }
        return;
    }

    ii = jj = 0;
}

void BlockSPADADriver::advanceElement()
{
    int i = bi*BS + ii;
    int j = bj*BS + jj;

    if (i >= M || j >= curCols) {
        jj++;
        if (jj == BS) { jj = 0; ii++; }
        if (ii == BS) advanceBlock();
        return;
    }

    accValue = 0.0f;
    remainingMul = innerDim;
    k = 0;
    sid = 0; // only one SID needed
    elemActive = true;
}

void BlockSPADADriver::issueMul()
{

    if (k >= innerDim)
        return;

    int pe = (bi*BS + ii) % numPEs;
    if (mulUnits[pe]->isFull())
        return;

    assert(curA && "curA is null");
    assert(curB && "curB is null");






    int i = bi*BS + ii;
    int j = bj*BS + jj;

    assert(i >= 0 && i < M && "i out of bounds");
    assert(j >= 0 && j < curCols && "j out of bounds");
    assert(k >= 0 && k < innerDim && "k out of bounds");

    float a = curA[i][k];
    float b = transposeB ? curB[j][k] : curB[k][j];

    mulUnits[pe]->push(a, b, sid);
    k++;
}

void BlockSPADADriver::onProductReady(int pe, float product, int)
{
    accUnits[pe]->push(product, sid);
}

void BlockSPADADriver::onAccReady(int, float sum, int)
{
    accValue += sum;
    remainingMul--;

    if (remainingMul == 0) {
        int i = bi*BS + ii;
        int j = bj*BS + jj;
        curOut[i][j] = accValue;

        elemActive = false;

        jj++;
        if (jj == BS) { jj = 0; ii++; }
        if (ii == BS) advanceBlock();
    }
}

void BlockSPADADriver::runSoftmax()
{
    for (int i = 0; i < M; i++) {
        float m = -INFINITY;
        for (int j = 0; j < N; j++)
            m = std::max(m, Scores[i][j]);

        float s = 0;
        for (int j = 0; j < N; j++)
            s += std::exp(Scores[i][j] - m);

        for (int j = 0; j < N; j++)
            Prob[i][j] = std::exp(Scores[i][j] - m) / s;
    }
}

void BlockSPADADriver::regStats()
{
    SimObject::regStats();
    stallCycles.name(name()+".stallCycles");
}

} // namespace gem5

#include "sparta/baseline/spada/AcceleratorDriverSPADA.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"

namespace gem5 {

AcceleratorDriverSPADA::
AcceleratorDriverSPADA(const AcceleratorDriverSPADAParams &p)
  : SimObject(p),
    Q((float**)p.Q), K((float**)p.K), V((float**)p.V),
    Scores((float**)p.Scores), Prob((float**)p.Prob),
    Output((float**)p.Output),
    M(p.M), N(p.N), Kdim(p.Kdim),
    numPEs(p.numPEs),
    startEvent([this]{ start(); }, "spada_start"),
    tickEvent([this]{ tick(); }, "spada_tick"),
    phase(PHASE_IDLE)
{
    for (auto *m : p.mul_units) mulUnits.push_back(m);
    for (auto *a : p.acc_units) accUnits.push_back(a);

    maxLiveOps = numPEs * (p.mul_queue_depth + p.acc_queue_depth + 1) * 4;

    sidToIdx.resize(maxLiveOps);
    sidRemaining.resize(maxLiveOps);
    sidPartial.resize(maxLiveOps);

    for (int i = 0; i < maxLiveOps; i++)
        freeSIDs.push(i);
}

void AcceleratorDriverSPADA::startup()
{
    for (int i = 0; i < numPEs; i++) {
        mulUnits[i]->setCallback(
            [this,i](float p,int sid){ onProductReady(i,p,sid); });
        accUnits[i]->setCallback(
            [this,i](float s,int sid){ onAccReady(i,s,sid); });
    }

    schedule(startEvent, curTick() + 1);
    schedule(tickEvent, curTick() + 1);
}

void AcceleratorDriverSPADA::start()
{
    phase = PHASE_QK;
    currentOut = Scores;
    currentCols = N;

    numReads  += M*Kdim + N*Kdim;
    numWrites += M*N;

    dispatchMatMulSPADA(Q, K, Scores, M, N, Kdim);
}

void AcceleratorDriverSPADA::buildSparseMeta(
    float **A, float **B, int M, int N, int K)
{
    A_csr.assign(M, {});
    B_csr.assign(N, {});

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            if (A[i][k] != 0) A_csr[i].cols.push_back(k);

    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            if (B[k][j] != 0) B_csr[j].cols.push_back(k);
}

void AcceleratorDriverSPADA::dispatchMatMulSPADA(
    float **A, float **B, float **C, int M, int N, int K)
{
    remaining.assign(M*N, 0);
    idxToSID.clear();
    mulQ = {};
    accQ = {};

    totalTasks = M * N;
    completedTasks = 0;

    buildSparseMeta(A, B, M, N, K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            int idx = i*N + j;
            int nz = 0;

            auto &ra = A_csr[i].cols;
            auto &cb = B_csr[j].cols;

            int pa = 0, pb = 0;
            while (pa < (int)ra.size() && pb < (int)cb.size()) {
                if (ra[pa] == cb[pb]) {
                    int k = ra[pa];
                    mulQ.push({A[i][k], B[k][j], idx});
                    nz++;
                    pa++; pb++;
                } else if (ra[pa] < cb[pb]) pa++;
                else pb++;
            }

            remaining[idx] = nz;
            if (nz == 0) {
                C[i][j] = 0.0f;
                completedTasks++;
            }
        }
    }

    tryScheduleMul();
}

void AcceleratorDriverSPADA::tryScheduleMul()
{
    if (mulQ.empty()) return;

    size_t qsz = mulQ.size();
    for (size_t it = 0; it < qsz; it++) {

        MulTask t = mulQ.front();
        bool issued = false;

        int sid = idxToSID.count(t.idx) ? idxToSID[t.idx] : -1;

        for (int pe = 0; pe < numPEs; pe++) {
            if (!mulUnits[pe]->isFull()) {

                if (sid == -1) {
                    if (freeSIDs.empty()) return;
                    sid = allocSID(t.idx, remaining[t.idx]);
                    idxToSID[t.idx] = sid;
                }

                mulUnits[pe]->push(t.a, t.b, sid);
                issued = true;
                break;
            }
        }

        mulQ.pop();
        if (!issued)
            mulQ.push(t);
    }
}

void AcceleratorDriverSPADA::onProductReady(int, float prod, int sid)
{
    accQ.push({prod, sid});
    tryScheduleAcc();
    tryScheduleMul();
}

void AcceleratorDriverSPADA::tryScheduleAcc()
{
    if (accQ.empty()) return;

    size_t qsz = accQ.size();
    for (size_t it = 0; it < qsz; it++) {

        AccTask t = accQ.front();
        bool issued = false;

        for (int pe = 0; pe < numPEs; pe++) {
            if (accUnits[pe]->push(t.val, t.sid)) {
                issued = true;
                break;
            }
        }

        accQ.pop();
        if (!issued)
            accQ.push(t);
    }
}

void AcceleratorDriverSPADA::onAccReady(int, float sum, int sid)
{
    sidPartial[sid] += sum;

    if (--sidRemaining[sid] == 0) {

        int idx = sidToIdx[sid];
        currentOut[idx/currentCols][idx%currentCols] = sidPartial[sid];
        completedTasks++;

        freeSID(sid);
        idxToSID.erase(idx);

        if (completedTasks == totalTasks) {
            if (phase == PHASE_QK) onQKDone();
            else if (phase == PHASE_AV) onAVDone();
        }
    }

    tryScheduleAcc();
    tryScheduleMul();
}

int AcceleratorDriverSPADA::allocSID(int idx, int nnz)
{
    int sid = freeSIDs.front();
    freeSIDs.pop();

    sidToIdx[sid] = idx;
    sidRemaining[sid] = nnz;
    sidPartial[sid] = 0.0f;

    return sid;
}

void AcceleratorDriverSPADA::freeSID(int sid)
{
    freeSIDs.push(sid);
}

void AcceleratorDriverSPADA::onQKDone()
{
    phase = PHASE_SOFTMAX;
    runSoftmax();
    startAV();
}

void AcceleratorDriverSPADA::runSoftmax()
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

    numReads  += M * N;
    numWrites += M * N;
}

void AcceleratorDriverSPADA::startAV()
{
    phase = PHASE_AV;
    currentOut = Output;
    currentCols = Kdim;

    numReads  += M*N + N*Kdim;
    numWrites += M*Kdim;

    dispatchMatMulSPADA(Prob, V, Output, M, Kdim, N);
}

void AcceleratorDriverSPADA::onAVDone()
{
    phase = PHASE_DONE;
    exitSimLoop("SPADA accelerator done");
}

void AcceleratorDriverSPADA::tick()
{
    if (phase != PHASE_DONE)
        schedule(tickEvent, curTick() + 1);
}

void AcceleratorDriverSPADA::regStats()
{
    SimObject::regStats();
    stallCycles.name(name() + ".stallCycles");
    numReads.name(name() + ".numReads");
    numWrites.name(name() + ".numWrites");
}

} // namespace gem5

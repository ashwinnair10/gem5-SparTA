#include "sparta/accelerator/AcceleratorDriver.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5 {

    AcceleratorDriver::
    AcceleratorDriver(const AcceleratorDriverParams &p)
        : SimObject(p),
        Q(reinterpret_cast<float **>(p.Q)),
        K(reinterpret_cast<float **>(p.K)),
        V(reinterpret_cast<float **>(p.V)),
        Scores(reinterpret_cast<float **>(p.Scores)),
        Prob(reinterpret_cast<float **>(p.Prob)),
        Output(reinterpret_cast<float **>(p.Output)),
        M(p.M), N(p.N), Kdim(p.Kdim),
        numPEs(p.numPEs),
        startEvent([this]{ start(); }, "start_event"),
        tickEvent([this]{tick();},"tick_event"),
        phase(PHASE_IDLE)
    {
        softmaxRow = new float[N];
        currentOut = nullptr;
        for (auto *m : p.mul_units){
            mulUnits.push_back(m);
        }
        for (auto *a : p.acc_units)
            accUnits.push_back(a);
        totalTasks = 0;
        completedTasks = 0;
        accBusy.resize(numPEs, -1);
        mulLoad.resize(numPEs,0);
        remaining.resize(M*N,0);

        maxLiveOps=numPEs*(p.mul_queue_depth+p.acc_queue_depth+1)*4;
        sidToIdx.resize(maxLiveOps);
        sidRemainingMul.resize(maxLiveOps);
        sidRemainingAcc.resize(maxLiveOps);
        sidPartialSum.resize(maxLiveOps);

        for (int i = 0; i < maxLiveOps; i++)
            freeSIDs.push(i);
    }

    void AcceleratorDriver::tick()
    {
        std::cout
            << "[SparTA-AccDriver] Tick : " << curTick() << "\n";
        if (phase == PHASE_DONE)
            return;

        bool mulBlocked =
            !mulTaskQueue.empty() &&
            std::all_of(mulUnits.begin(), mulUnits.end(),
                [](auto *m){ return m->isFull(); });

        bool accBlocked =
            !accTaskQueue.empty() &&
            std::all_of(accUnits.begin(), accUnits.end(),
                [](auto *a){ return a->isFull(); });

        if (mulBlocked || accBlocked)
            stallCycles++;

        schedule(tickEvent, curTick() + 1);
    }


    void AcceleratorDriver::startup()
    {
        std::cout << RED
            << "[SparTA-Acc] startup with " << numPEs << " PEs\n"<< RESET;
        for (int i = 0; i < numPEs; i++) {
            mulUnits[i]->setCallback(
                [this, i](float product,int sid){
                    this->onProductReady(i, product,sid);
                });
            accUnits[i]->setCallback(
                [this, i](float sum,int sid){
                    this->onAccReady(i, sum,sid);
                });
        }
        schedule(startEvent, curTick() + 1);
        schedule(tickEvent, curTick() + 1);
    }

    void AcceleratorDriver::start()
    {
        std::cout << RED<< "[SparTA-Accl] Starting Q*K^T \n"<< RESET;
        phase = PHASE_QK;
        currentOut=Scores;
        currentCols=N;
        numReads  += M * Kdim;
        numReads  += N * Kdim;
        numWrites += M * N;

        dispatchMatMul(Q, K, Scores, M, N, Kdim);
    }

    void
    AcceleratorDriver::dispatchMatMul(float **A, float **B, float **C,
                                        int M, int N, int K)
    {
        remaining.clear();
        remaining.resize(M*N);
        assert(freeSIDs.size() == maxLiveOps);

        mulTaskQueue = std::queue<MulTask>();
        accTaskQueue = std::queue<AccTask>();
        sidToAccPE.clear();

        totalTasks = M * N;
        completedTasks = 0;
        int nz=0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i * N + j;
                nz = 0;

                for (int k = 0; k < K; k++) {
                    if (A[i][k] != 0 && B[k][j] != 0){
                        nz++;
                        mulTaskQueue.push({A[i][k], B[k][j], idx});
                    }
                }

                remaining[idx]=nz;

                if (nz == 0) {
                    currentOut[i][j] = 0;
                    completedTasks++;
                    continue;
                }
            }
        }
        tryScheduleMul();
    }

    void AcceleratorDriver::tryScheduleMul()
    {
        if (mulTaskQueue.empty()) return ;
        size_t qsz = mulTaskQueue.size();

        for (size_t it = 0; it < qsz; it++) {
            MulTask t = mulTaskQueue.front();

            bool issued = false;

            auto sid=idxToSID.find(t.idx)==idxToSID.end()?
                 -1 : idxToSID[t.idx];

            int home = sid!=-1?hashToPE(sid): hashToPE(t.idx);

            for (int off = 0; off < numPEs; off++) {
                int pe = (home + off) % numPEs;
                if (!mulUnits[pe]->isFull()){
                    if (freeSIDs.empty())
                        return;
                    if (sid==-1){
                        sid=allocSID(t.idx,remaining[t.idx]);
                        idxToSID[t.idx]=sid;
                    }
                    if (mulUnits[pe]->push(t.a, t.b,sid)){
                        mulLoad[pe]++;
                        issued = true;
                        break;
                    }
                }
            }
            mulTaskQueue.pop();
            if (issued){
                continue;
            }
            else{
                mulTaskQueue.push(t);
            }
        }
        return ;
    }



    void AcceleratorDriver::onProductReady(int pe, float product,int sid)
    {
        mulLoad[pe]--;
        sidRemainingMul[sid]--;
        accTaskQueue.push({product,sid});
        tryScheduleAcc();
        tryScheduleMul();

    }


    void AcceleratorDriver::tryScheduleAcc()
    {
        if (accTaskQueue.empty())
            return;

        size_t qsz = accTaskQueue.size();

        for (size_t it = 0; it < qsz; it++) {

            AccTask t = accTaskQueue.front();
            accTaskQueue.pop();

            bool issued = false;

            int home = hashToPE(t.sid);

            for (int off = 0; off < numPEs; off++) {
                int pe = (home + off) % numPEs;

                if (accUnits[pe]->push(t.product, t.sid)) {
                    // std::cout << RED
                    //     << "[SparTA-AccDriver] Scheduled Acc Task SID: "
                    //     << t.sid << " on PE: " << pe << "\n" << RESET;
                    issued = true;
                    break;
                }
            }

            if (!issued) {
                accTaskQueue.push(t);
            }
        }
    }




    int AcceleratorDriver::findFreeAccPE(int sid)
    {
        for (int i = 0; i < numPEs; i++) {
            if (accBusy[i] == sid)
                return i;
        }

        int pe = hashToPE(sid);

        if (accBusy[pe] == -1)
            return pe;

        for (int offset = 1; offset < numPEs; offset++) {
            int probe = (pe + offset) % numPEs;
            if (accBusy[probe] == -1)
                return probe;
        }

        return -1;
    }

    void AcceleratorDriver::reseed()
    {
        int maxLoad = 0, minLoad = INT_MAX;
        for (int l : mulLoad) {
            maxLoad = std::max(maxLoad, l);
            minLoad = std::min(minLoad, l);
        }

        if (maxLoad - minLoad > 2) {
            hashSeed++;
        }
    }


    void AcceleratorDriver::
    onAccReady(int pe, float sum,int sid)
    {
        sidRemainingAcc[sid]--;
        sidPartialSum[sid] += sum;
        // std::cout << RED
        //     << "[SparTA-AccDriver] Acc Task SID: "
        //     << sid << " completed on PE: " << pe
        //     << " -- Partial Sum: " << sum
        //     << " -- Remaining Ops: " << sidRemainingAcc[sid]
        //     << "\n" << RESET;
        if (sidRemainingAcc[sid] == 0) {

            int idx = sidToIdx[sid];
            int i = idx / currentCols;
            int j = idx % currentCols;

            currentOut[i][j] = sidPartialSum[sid];

            completedTasks++;

            freeSID(sid);
            idxToSID.erase(idx);

            if (completedTasks == totalTasks) {
                if (phase == PHASE_QK)
                    onQKDone();
                else if (phase == PHASE_AV)
                    onAVDone();
            }

        }
        if ((completedTasks & 0x3F) == 0) {
            reseed();
        }
        tryScheduleAcc();
    }

    void AcceleratorDriver::onQKDone()
    {
        std::cout << RED<< "[SparTA-Accl] QK done. Running softmax.\n"<< RESET;
        phase = PHASE_SOFTMAX;
        runSoftmax();
        startAV();
    }

    void AcceleratorDriver::runSoftmax()
    {
        for (int i = 0; i < M; i++) {
            float m = -INFINITY;
            for (int j = 0; j < N; j++)
                m = std::max(m, Scores[i][j]);
            float s = 0;
            for (int j = 0; j < N; j++) {
                softmaxRow[j] = std::exp(Scores[i][j] - m);
                s += softmaxRow[j];
            }
            for (int j = 0; j < N; j++)
                Prob[i][j] = softmaxRow[j] / s;
        }
        numReads  += M * N;
        numWrites += M * N;

    }

    void AcceleratorDriver::startAV()
    {
        phase = PHASE_AV;
        std::cout << RED<< "[SparTA-Accl] Starting A*V\n"<< RESET;
        currentOut=Output;
        currentCols=Kdim;
        numReads  += M * N;
        numReads  += N * Kdim;
        numWrites += M * Kdim;

        dispatchMatMul(Prob, V, Output, M, Kdim, N);
    }

    void AcceleratorDriver::onAVDone()
    {
        std::cout << RED
            << "[SparTA-Accl] AV done. Accelerator complete.\n"<< RESET;
        phase = PHASE_DONE;
        exitSimLoop("Accelerator done");
    }

    void AcceleratorDriver::regStats()
    {
        using namespace statistics;
        SimObject::regStats();
        numReads
            .name(name() + ".numReads")
            .desc("Number of reads performed by the accelerator driver");
        numWrites
            .name(name() + ".numWrites")
            .desc("Number of writes performed by the accelerator driver");
        stallCycles
            .name(name() + ".stallCycles")
            .desc("Number of cycles the accelerator driver was stalled");
    }

    int AcceleratorDriver::allocSID(int idx,int nnz){
        assert(!freeSIDs.empty());

        int sid = freeSIDs.front();
        freeSIDs.pop();

        sidToIdx[sid] = idx;
        sidRemainingAcc[sid] = nnz;
        sidRemainingMul[sid] = nnz;
        sidPartialSum[sid] = 0.0f;

        return sid;
    }

    void AcceleratorDriver::freeSID(int sid){
        freeSIDs.push(sid);
        tryScheduleMul();
        tryScheduleAcc();
    }
}

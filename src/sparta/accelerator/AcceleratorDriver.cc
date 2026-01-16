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
        startEvent([this]{ start(); }, "baseline_start_event"),
        retryPending(false),
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
        mulBusy.resize(numPEs, -1);
        accBusy.resize(numPEs, -1);
        accLoad.resize(numPEs, 0);
    }

    void AcceleratorDriver::startup()
    {
        std::cout << RED
            << "[SparTA-Acc] startup with " << numPEs << " PEs\n"<< RESET;
        for (int i = 0; i < numPEs; i++) {
            mulUnits[i]->setCallback(
                [this, i](float product,int idx){
                    this->onProductReady(i, product,idx);
                });
            accUnits[i]->setCallback(
                [this, i](float sum, int remaining,int idx){
                    this->onAccReady(i, sum, remaining,idx);
                });
        }
        schedule(startEvent, curTick() + 1);
    }

    void AcceleratorDriver::start()
    {
        std::cout << RED<< "[SparTA-Accl] Starting Q*K^T \n"<< RESET;
        phase = PHASE_QK;
        currentOut=Scores;
        currentCols=N;
        dispatchMatMul(Q, K, Scores, M, N, Kdim);
    }

    void
    AcceleratorDriver::dispatchMatMul(float **A, float **B, float **C,
                                        int M, int N, int K)
    {
        partialSums.clear();
        remainingCounts.clear();
        mulTaskQueue = std::queue<MulTask>();

        totalTasks = M * N;
        completedTasks = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i * N + j;
                remainingCounts[idx] = K;
                partialSums[idx] = 0.0f;
                for (int k = 0; k < K; k++) {
                    numReads += 2; // read A[i][k] and B[k][j]
                    if (A[i][k]==0.0f || B[k][j]==0.0f) {
                        remainingCounts[idx]--;
                        continue;
                    }
                    mulTaskQueue.push({A[i][k], B[k][j], idx});
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
            int home = hashToPE(t.idx);

            for (int off = 0; off < numPEs; off++) {
                int pe = (home + off) % numPEs;

                if (mulBusy[pe] == -1) {
                    if (mulUnits[pe]->push(t.a, t.b, t.idx)) {
                        mulBusy[pe] = t.idx;
                        issued = true;
                        break;
                    }
                }
            }
            if (issued){
                mulTaskQueue.pop();
                continue;
            }
        }
        return ;
    }



    void AcceleratorDriver::onProductReady(int pe, float product,int idx)
    {
        mulBusy[pe] = -1;

        accTaskQueue.push({product,idx});
        tryScheduleAcc();
        tryScheduleMul();

    }

    void AcceleratorDriver::tryScheduleAcc()
    {

        if (accTaskQueue.empty()) return ;

        size_t qsz = accTaskQueue.size();

        for (size_t it = 0; it < qsz; it++) {
            AccTask t = accTaskQueue.front();
            bool issued = false;
            if (idxToAccPE.find(t.idx)!=idxToAccPE.end()){
                int pe = idxToAccPE[t.idx];
                    accUnits[pe]->setParams(
                        partialSums[t.idx],
                        remainingCounts[t.idx]
                    );
                    if (accUnits[pe]->push(t.product, t.idx)) {
                        accBusy[pe] = t.idx;
                        accLoad[pe]++;
                        issued = true;
                    }
            }
            else{
                int home = findFreeAccPE(t.idx);

                if (home == -1){
                    break;
                }

                for (int off = 0; off < numPEs; off++) {
                    int pe = (home + off) % numPEs;

                    if (accBusy[pe] == -1) {
                        accUnits[pe]->setParams(
                            partialSums[t.idx],
                            remainingCounts[t.idx]
                        );
                        if (accUnits[pe]->push(t.product, t.idx)) {
                            accBusy[pe] = t.idx;
                            accLoad[pe]++;
                            idxToAccPE[t.idx]=pe;
                            issued = true;
                            break;
                        }
                    }
                }
            }
            if (issued){
                accTaskQueue.pop();
                continue;
            }
        }
        return ;
    }


    int AcceleratorDriver::findFreeAccPE(int idx)
    {
        for (int i = 0; i < numPEs; i++) {
            if (accBusy[i] == idx)
                return i;
        }

        int pe = hashToPE(idx);

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
        for (int l : accLoad) {
            maxLoad = std::max(maxLoad, l);
            minLoad = std::min(minLoad, l);
        }

        if (maxLoad - minLoad > 2) {
            hashSeed++;
        }
    }




    void AcceleratorDriver::
    onAccReady(int pe, float sum, int remaining,int idx)
    {
        if (remaining==0){
            currentOut[idx/currentCols][idx%currentCols]=sum;
            numWrites++; // write output
            completedTasks++;
            partialSums[idx]=0.0f;
            remainingCounts[idx]=(phase==PHASE_QK?Kdim:N);
            accUnits[pe]->reset(phase==PHASE_QK?Kdim:N);
            accLoad[pe]--;
            accBusy[pe]=-1;
            idxToAccPE.erase(idx);
            reseed();
            if (completedTasks == totalTasks) {
                if (phase == PHASE_QK) {
                    onQKDone();
                }
                else if (phase == PHASE_AV) {
                    onAVDone();
                }
            }
        }else{
            partialSums[idx]=sum;
            remainingCounts[idx]=remaining;
            accBusy[pe]=idx;
            accLoad[pe]--;
            idxToAccPE[idx]=pe;
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
    }

    void AcceleratorDriver::startAV()
    {
        phase = PHASE_AV;
        std::cout << RED<< "[SparTA-Accl] Starting A*V\n"<< RESET;
        currentOut=Output;
        currentCols=Kdim;
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
        numReads
            .name(name() + ".numReads")
            .desc("Number of reads performed by the accelerator driver");
        numWrites
            .name(name() + ".numWrites")
            .desc("Number of writes performed by the accelerator driver");
        stallCycles
            .name(name() + ".stallCycles")
            .desc("Number of cycles the accelerator driver was stalled");
        for (size_t i=0;i<mulUnits.size();i++){
            if (mulUnits[i])
                mulUnits[i]->regStats();
        }
        for (size_t i=0;i<accUnits.size();i++){
            if (accUnits[i])
                accUnits[i]->regStats();
        }
    }
}

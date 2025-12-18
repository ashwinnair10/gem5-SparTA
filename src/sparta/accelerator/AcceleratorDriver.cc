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

    //@TODO: add hashing to a PE to do mul instead of
    // searching through all pe to find free
    void AcceleratorDriver::tryScheduleMul()
    {
        for (int pe = 0; pe < numPEs; pe++) {
            if (mulBusy[pe] == -1 && !mulTaskQueue.empty()) {
                MulTask t = mulTaskQueue.front();
                mulTaskQueue.pop();

                mulBusy[pe] = t.idx;

                mulUnits[pe]->startCompute(t.a, t.b, t.idx);
            }
        }
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
        if (accTaskQueue.empty()) return;

        AccTask t = accTaskQueue.front();
        int pe = findFreeAccPE(t.idx);

        if (pe == -1) return;   // no free PE, backpressure

        accTaskQueue.pop();

        if (accBusy[pe] == -1) {
            accBusy[pe] = t.idx;
            accLoad[pe]++;

            accUnits[pe]->setParams(
                partialSums[t.idx],
                remainingCounts[t.idx]
            );
        }

        accUnits[pe]->feedProduct(t.product, t.idx);
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

        for (int offset = 1; offset <= 2; offset++) {
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
            completedTasks++;
            partialSums[idx]=0.0f;
            remainingCounts[idx]=(phase==PHASE_QK?Kdim:N);
            accUnits[pe]->reset(phase==PHASE_QK?Kdim:N);
            accLoad[pe]--;
            accBusy[pe]=-1;
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
}

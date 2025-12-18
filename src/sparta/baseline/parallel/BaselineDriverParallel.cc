#include "sparta/baseline/parallel/BaselineDriverParallel.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5 {

    BaselineDriverParallel::
    BaselineDriverParallel(const BaselineDriverParallelParams &p)
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
    }

    void BaselineDriverParallel::startup()
    {
        std::cout << "[SparTA-BASE] startup with " << numPEs << " PEs\n";
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

    void BaselineDriverParallel::start()
    {
        std::cout << "[SparTA-BASE] Starting Q*K^T PARALLEL\n";
        phase = PHASE_QK;
        currentOut=Scores;
        currentCols=N;
        dispatchMatMul(Q, K, Scores, M, N, Kdim);
    }

    void
    BaselineDriverParallel::dispatchMatMul(float **A, float **B, float **C,
                                        int M, int N, int K)
    {
        for (auto acc:accUnits){
            acc->reset(K);
        }
        totalTasks = M * N;
        completedTasks = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < K; k++) {
                    int mul_id = (i*N+j) % numPEs;
                    float a = A[i][k];
                    float b = B[k][j];
                    mulUnits[mul_id]->startCompute(a, b,i*N+j);
                }
            }
        }
    }

    void BaselineDriverParallel::onProductReady(int pe, float product,int idx)
    {
        accUnits[pe]->feedProduct(product,idx);
    }

    void BaselineDriverParallel::
    onAccReady(int pe, float sum, int remaining,int idx)
    {
        if (remaining==0){
            currentOut[idx/currentCols][idx%currentCols]=sum;
            completedTasks++;
            accUnits[pe]->reset(phase==PHASE_QK?Kdim:N);
            if (completedTasks == totalTasks) {
                if (phase == PHASE_QK) {
                    onQKDone();
                }
                else if (phase == PHASE_AV) {
                    onAVDone();
                }
            }
        }
    }

    void BaselineDriverParallel::onQKDone()
    {
        std::cout << "[SparTA-BASE] QK done. Running softmax.\n";
        phase = PHASE_SOFTMAX;
        runSoftmax();
        startAV();
    }

    void BaselineDriverParallel::runSoftmax()
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

    void BaselineDriverParallel::startAV()
    {
        phase = PHASE_AV;
        std::cout << "[SparTA-BASE] Starting A*V\n";
        currentOut=Output;
        currentCols=Kdim;
        dispatchMatMul(Prob, V, Output, M, Kdim, N);
    }

    void BaselineDriverParallel::onAVDone()
    {
        std::cout << "[SparTA-BASE] AV done. Baseline complete.\n";
        phase = PHASE_DONE;
        exitSimLoop("baseline done");
    }
}

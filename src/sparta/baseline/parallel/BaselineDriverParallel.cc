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
        retryEvent([this]{ retryStalled(); }, "baseline_retry_event"),
        tickEvent([this]{ tick(); }, "baseline_tick_event"),
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

    void BaselineDriverParallel::tick(){
        if (phase == PHASE_DONE)
        return;

        bool stalled =
            !stallMulQueue.empty() ||
            !stallAccQueue.empty();

        if (stalled)
            stallCycles++;

        schedule(tickEvent, curTick() + 1);
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
                [this, i](float sum,int idx){
                    this->onAccReady(i, sum,idx);
                });
        }
        schedule(startEvent, curTick() + 1);
        schedule(tickEvent, curTick() + 1);
    }

    void BaselineDriverParallel::start()
    {
        std::cout << "[SparTA-BASE] Starting Q*K^T PARALLEL\n";
        phase = PHASE_QK;
        currentOut=Scores;
        currentCols=N;
        numReads  += M * Kdim;
        numReads  += N * Kdim;
        numWrites += M * N;

        dispatchMatMul(Q, K, Scores, M, N, Kdim);
    }

    void
    BaselineDriverParallel::dispatchMatMul(float **A, float **B, float **C,
                                        int M, int N, int K)
    {
        partialSums.assign(M * N, 0.0f);
        remainingCounts.assign(M * N, K);

        totalTasks = M * N;
        completedTasks = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < K; k++) {
                    int mul_id = (i*N+j) % numPEs;
                    float a = A[i][k];
                    float b = B[k][j];
                    if (!mulUnits[mul_id]->push(a, b,i*N+j)){
                        stallMulQueue.push(std::make_tuple(a,b,i*N+j));
                        if (!retryEvent.scheduled())
                            schedule(retryEvent, curTick() + 1);
                    }
                }
            }
        }
    }

    void BaselineDriverParallel::onProductReady(int pe, float product,int idx)
    {
        if (!accUnits[pe]->push(product,idx)){
            stallAccQueue.push(std::make_pair(product,idx));
            if (!retryEvent.scheduled())
                schedule(retryEvent, curTick() + 1);
            return;
        }
    }

    void BaselineDriverParallel::
    onAccReady(int pe, float sum,int idx)
    {
        partialSums[idx] += sum;
        remainingCounts[idx]--;
        if (remainingCounts[idx]==0){
            currentOut[idx/currentCols][idx%currentCols]=partialSums[idx];
            completedTasks++;
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

    void BaselineDriverParallel::retryStalled()
    {
        bool stalled = false;

        size_t mul_sz = stallMulQueue.size();
        for (size_t i = 0; i < mul_sz; i++) {
            auto [a, b, idx] = stallMulQueue.front();
            stallMulQueue.pop();

            int mul_id = idx % numPEs;
            if (!mulUnits[mul_id]->push(a, b, idx)) {
                stallMulQueue.push({a, b, idx});
                stalled = true;
            }
        }

        size_t acc_sz = stallAccQueue.size();
        for (size_t i = 0; i < acc_sz; i++) {
            auto [product, idx] = stallAccQueue.front();
            stallAccQueue.pop();

            int acc_id = idx % numPEs;
            if (!accUnits[acc_id]->push(product, idx)) {
                stallAccQueue.push({product, idx});
                stalled = true;
            }
        }

        if (stalled || !stallMulQueue.empty() || !stallAccQueue.empty()) {
            schedule(retryEvent, curTick() + 1);
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
        numReads  += M * N;
        numWrites += M * N;

    }

    void BaselineDriverParallel::startAV()
    {
        phase = PHASE_AV;
        std::cout << "[SparTA-BASE] Starting A*V\n";
        currentOut=Output;
        currentCols=Kdim;
        numReads  += M * N;
        numReads  += N * Kdim;
        numWrites += M * Kdim;

        dispatchMatMul(Prob, V, Output, M, Kdim, N);
    }

    void BaselineDriverParallel::onAVDone()
    {
        std::cout << "[SparTA-BASE] AV done. Baseline complete.\n";
        phase = PHASE_DONE;
        exitSimLoop("baseline done");
    }

    void BaselineDriverParallel::regStats()
    {
        using namespace statistics;
        SimObject::regStats();
        numReads
            .name(name() + ".numReads")
            .desc("Number of reads performed by the baseline driver");
        numWrites
            .name(name() + ".numWrites")
            .desc("Number of writes performed by the baseline driver");
        stallCycles
            .name(name() + ".stallCycles")
            .desc("Number of cycles stalled due to full PE queues");
    }
}

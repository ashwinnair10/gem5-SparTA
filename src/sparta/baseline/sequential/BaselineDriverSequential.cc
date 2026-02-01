#include "sparta/baseline/sequential/BaselineDriverSequential.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"

namespace gem5 {

    BaselineDriverSequential::
    BaselineDriverSequential(const BaselineDriverSequentialParams &p)
        : SimObject(p),
        mm(p.mm),
        Q(reinterpret_cast<float **>(p.Q)),
        K(reinterpret_cast<float **>(p.K)),
        V(reinterpret_cast<float **>(p.V)),
        Scores(reinterpret_cast<float **>(p.Scores)),
        Prob(reinterpret_cast<float **>(p.Prob)),
        Output(reinterpret_cast<float **>(p.Output)),
        M(p.M), N(p.N), Kdim(p.Kdim),
        startEvent([this]{ start(); }, "baseline_start_event"),
        tickEvent([this]{ tick(); }, "baseline_tick_event"),
        phase(PHASE_IDLE)
    {
        softmaxRow = new float[N];
    }

    void BaselineDriverSequential::tick()
    {
        if (phase == PHASE_DONE)
            return;
        if (phase == PHASE_QK || phase == PHASE_AV)
            stallCycles++;

        schedule(tickEvent, curTick() + 1);
    }

    void BaselineDriverSequential::startup()
    {
        std::cout << "[SparTA-BASE] startup\n";
        mm->setFinishedCallback([this](){
            if (phase == PHASE_QK)
                onQKDone();
            else if (phase == PHASE_AV)
                onAVDone();
        });
        schedule(startEvent, curTick() + 1);
        schedule(tickEvent, curTick() + 1);
    }

    void BaselineDriverSequential::start()
    {
        phase = PHASE_QK;
        std::cout << "[SparTA-BASE] Starting Q*K^T\n";
        mm->startMatMul(
            (uint64_t)Q,
            (uint64_t)K,
            (uint64_t)Scores,
            M, N, Kdim
        );
        numReads  += M * Kdim;
        numReads  += N * Kdim;
        numWrites += M * N;
    }

    void BaselineDriverSequential::onQKDone()
    {
        std::cout << "[SparTA-BASE] Q*K^T complete. Running softmax.\n";
        phase = PHASE_SOFTMAX;
        runSoftmax();
        startAV();
    }

    void BaselineDriverSequential::runSoftmax()
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
        std::cout << "[SparTA-BASE] Softmax done.\n";
    }

    void BaselineDriverSequential::startAV()
    {
        std::cout << "[SparTA-BASE] Starting A*V\n";
        phase = PHASE_AV;
        mm->startMatMul(
            (uint64_t)Prob,
            (uint64_t)V,
            (uint64_t)Output,
            M, Kdim, N
        );
        numReads  += M * N;
        numReads  += N * Kdim;
        numWrites += M * Kdim;
    }

    void BaselineDriverSequential::onAVDone()
    {
        std::cout << "[SparTA-BASE] AV complete. Baseline Attention Done.\n";
        phase = PHASE_DONE;
        exitSimLoop("");
    }

    void BaselineDriverSequential::regStats()
    {
        using namespace statistics;
        SimObject::regStats();

        numReads
            .name(name() + ".num_reads")
            .desc("Number of reads performed by the Baseline Driver")
            ;

        numWrites
            .name(name() + ".num_writes")
            .desc("Number of writes performed by the Baseline Driver")
            ;

        stallCycles
            .name(name() + ".stall_cycles")
            .desc("Number of cycles the Baseline Driver was stalled")
            ;
    }

}

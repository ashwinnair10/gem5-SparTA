#include "mem/ruby/sparta/baseline/sequential/BaselineDriver.hh"
#include <cmath>
#include <iostream>
#include "sim/sim_exit.hh"

namespace gem5 {

BaselineDriver::BaselineDriver(const BaselineDriverParams &p)
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
      phase(PHASE_IDLE)
{
    softmaxRow = new float[N];
}

void
BaselineDriver::startup()
{
    std::cout << "[SparTA-BASELINEDRIVER] startup\n";

    mm->setFinishedCallback([this](){
        if (phase == PHASE_QK)
            onQKDone();
        else if (phase == PHASE_AV)
            onAVDone();
    });

    schedule(startEvent, curTick() + 1);
}

void
BaselineDriver::start()
{
    phase = PHASE_QK;
    std::cout << "[SparTA-BASELINEDRIVER] Starting Q*K^T\n";

    mm->startMatMul(
        (uint64_t)Q,
        (uint64_t)K,
        (uint64_t)Scores,
        M, N, Kdim
    );
}

void
BaselineDriver::onQKDone()
{
    std::cout << "[SparTA-BASELINEDRIVER] Q*K^T complete. Running softmax.\n";

    phase = PHASE_SOFTMAX;

    runSoftmax();

    startAV();
}

void
BaselineDriver::runSoftmax()
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

    std::cout << "[SparTA-BASELINEDRIVER] Softmax done.\n";
}

void
BaselineDriver::startAV()
{
    std::cout << "[SparTA-BASELINEDRIVER] Starting A*V\n";

    phase = PHASE_AV;

    mm->startMatMul(
        (uint64_t)Prob,
        (uint64_t)V,
        (uint64_t)Output,
        M, Kdim, N
    );
}

void
BaselineDriver::onAVDone()
{
    std::cout << "[SparTA-BASELINEDRIVER] AV complete. Baseline Attention Done.\n";

    phase = PHASE_DONE;
    exitSimLoop("");
}

}

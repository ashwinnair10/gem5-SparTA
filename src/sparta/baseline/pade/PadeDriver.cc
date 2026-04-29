#include "sparta/baseline/pade/PadeDriver.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5
{

PadeDriver::PadeDriver(const PadeDriverParams &p)
    : SimObject(p),
      X(reinterpret_cast<float **>(p.X)),
      WQ(reinterpret_cast<float **>(p.WQ)),
      WK(reinterpret_cast<float **>(p.WK)),
      WV(reinterpret_cast<float **>(p.WV)),
      projPart(0),
      Q(reinterpret_cast<float **>(p.Q)),
      K(reinterpret_cast<float **>(p.K)),
      V(reinterpret_cast<float **>(p.V)),
      Scores(reinterpret_cast<float **>(p.Scores)),
      Prob(reinterpret_cast<float **>(p.Prob)),
      Output(reinterpret_cast<float **>(p.Output)),
      M(p.M),
      N(p.N),
      Kdim(p.Kdim),
      numPEs(p.numPEs),
      startEvent([this] { start(); }, "baseline_start_event"),
      retryEvent([this] { retryStalled(); }, "baseline_retry_event"),
      tickEvent([this] { tick(); }, "baseline_tick_event"),
      phase(PHASE_IDLE)
{
    softmaxRow = new float[N];
    currentOut = nullptr;
    for (auto *m : p.mul_units) {
        mulUnits.push_back(m);
    }
    for (auto *a : p.acc_units) {
        accUnits.push_back(a);
    }
    totalTasks = 0;
    completedTasks = 0;
    isTranspose = false;
}

void
PadeDriver::tick()
{
    if (phase == PHASE_DONE) {
        return;
    }
    bool stalled = !stallMulQueue.empty() || !stallAccQueue.empty();

    if (stalled) {
        stallCycles++;
    }

    schedule(tickEvent, curTick() + 1);
}

void
PadeDriver::startup()
{
    std::cout << "[Parallel] startup with " << numPEs << " PEs\n";
    for (int i = 0; i < numPEs; i++) {
        mulUnits[i]->setCallback([this, i](float product, int idx) {
            this->onProductReady(i, product, idx);
        });
        accUnits[i]->setCallback(
            [this, i](float sum, int idx) { this->onAccReady(i, sum, idx); });
    }
    schedule(startEvent, curTick() + 1);
    schedule(tickEvent, curTick() + 1);
}

void
PadeDriver::start()
{
    std::cout << "[Parallel] Starting projection (parallel)\n";
    phase = PHASE_PROJ;
    startProjection();
}

void
PadeDriver::startProjection()
{
    float **out;
    float **weight;

    if (projPart == 0) {
        out = Q;
        weight = WQ;
    } else if (projPart == 1) {
        out = K;
        weight = WK;
    } else {
        out = V;
        weight = WV;
    }

    currentOut = out;
    currentCols = Kdim;
    numReads += M * Kdim;
    numReads += Kdim * Kdim;
    numWrites += M * Kdim;
    isTranspose = false;

    dispatchMatMul(X, weight, out, M, Kdim, Kdim);
}

void
PadeDriver::onProjectionDone()
{
    projPart++;
    std::cout << "[Parallel] Projection part " << projPart << " done.\n";

    if (projPart < 3) {
        startProjection();
        return;
    }

    std::cout << "[Parallel] Projection done. Starting QK\n";

    phase = PHASE_QK;
    currentOut = Scores;
    currentCols = N;

    numReads += M * Kdim;
    numReads += N * Kdim;
    numWrites += M * N;
    isTranspose = true;
    dispatchMatMul(Q, K, Scores, M, N, Kdim, true);
}

void
PadeDriver::onAccReady(int pe, float sum, int idx)
{
    auto &state = padeTracker[idx];

    state.total_dot_prod += sum;
    state.current_k++;

    if (state.current_k < currentK && !state.is_pruned) {
        state.current_bit_round = 0;
        state.partial_sum = 0.0f;
        issueBit(idx, pe);
    } else {
        handleTaskCompletion(idx);
    }
}

void
PadeDriver::retryStalled()
{
    bool stalled = false;

    size_t mul_sz = stallMulQueue.size();
    for (size_t i = 0; i < mul_sz; i++) {
        auto [a, b, idx] = stallMulQueue.front();
        stallMulQueue.pop();

        int mul_id = findLeastLoadedPE();
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

void
PadeDriver::onQKDone()
{
    std::cout << "[Parallel] QK done. Running softmax.\n";
    phase = PHASE_SOFTMAX;
    runSoftmax();
    startAV();
}

void
PadeDriver::runSoftmax()
{
    for (int i = 0; i < M; i++) {
        float m = -INFINITY;
        for (int j = 0; j < N; j++) {
            m = std::max(m, Scores[i][j]);
        }
        float s = 0;
        for (int j = 0; j < N; j++) {
            softmaxRow[j] = std::exp(Scores[i][j] - m);
            s += softmaxRow[j];
        }
        for (int j = 0; j < N; j++) {
            Prob[i][j] = softmaxRow[j] / s;
        }
    }
    numReads += M * N;
    numWrites += M * N;
}

void
PadeDriver::startAV()
{
    phase = PHASE_AV;
    std::cout << "[Parallel] Starting A*V\n";
    currentOut = Output;
    currentCols = Kdim;
    numReads += M * N;
    numReads += N * Kdim;
    numWrites += M * Kdim;
    isTranspose = false;

    dispatchMatMul(Prob, V, Output, M, Kdim, N);
}

void
PadeDriver::onAVDone()
{
    std::cout << "[Parallel] AV done. Baseline complete.\n";
    phase = PHASE_DONE;
    exitSimLoop("baseline done");
}

void
PadeDriver::dispatchMatMul(float **A, float **B, float **C, int _M, int _N,
                           int _K, bool transpose)
{
    currentA = A;
    currentB = B;
    currentOut = C;
    currentK = _K;
    padeTracker.clear();
    rowMax.assign(_M, 0.0f);
    completedTasks = 0;
    totalTasks = _M * _N;
    isTranspose = transpose;
    activeTasks.clear();
    std::cout << "STARTING MATMUL" << '\n';

    for (int i = 0; i < _M; i++) {
        for (int k = 0; k < _K; k++) {
            rowMax[i] = std::max(rowMax[i], std::abs(A[i][k]));
        }
    }

    for (int i = 0; i < _M; i++) {
        for (int j = 0; j < _N; j++) {
            int idx = i * _N + j;
            padeTracker[idx] = PadeBitState(0, 0, 0.0f, 0.0f, false);
            activeTasks.push_back(idx);
        }
    }

    tryDispatchNext();
}

void
PadeDriver::tryDispatchNext()
{
    if (activeTasks.empty()) {
        return;
    }
    int attempts = activeTasks.size();
    while (attempts--) {
        int idx = pickBestTask();
        if (idx == -1) {
            return;
        }
        int pe = findLeastLoadedPE();

        if (mulUnits[pe]->isFull()) {
            return;
        }
        issueBit(idx, pe);
    }
}

int
PadeDriver::findLeastLoadedPE()
{
    int best = 0;
    int minLoad = INT_MAX;

    for (int i = 0; i < numPEs; i++) {
        int load = mulUnits[i]->queueSize();
        if (load < minLoad) {
            minLoad = load;
            best = i;
        }
    }
    return best;
}

int
PadeDriver::pickBestTask()
{
    if (activeTasks.empty()) {
        return -1;
    }
    int r = rand() % activeTasks.size();
    return activeTasks[r];
}

void
PadeDriver::issueBit(int idx, int pe)
{
    if (idx < 0 || idx >= totalTasks) {
        return;
    }

    auto &state = padeTracker[idx];

    if (state.current_k >= currentK) {
        return;
    }

    int i = idx / currentCols, j = idx % currentCols, k = state.current_k;
    if (idx < 5) {
        std::cout << "[ISSUE] idx=" << idx << " i=" << i << " j=" << j
                  << " k=" << k << " currentCols=" << currentCols
                  << " currentK=" << currentK << std::endl;
    }
    float a = std::max(-1.0f, std::min(1.0f, currentA[i][k]));
    int act_val = (int)std::round(a * 127);
    int bit = (act_val >> (7 - state.current_bit_round)) & 0x1;
    float bit_slice = (float)bit;
    float weight = isTranspose ? currentB[j][k] : currentB[k][j];
    if (idx < 5) {
        std::cout << "[WEIGHT] " << (isTranspose ? "B[j][k]" : "B[k][j]")
                  << " = " << weight << std::endl;
    }
    if (idx < 5) {
        std::cout << "[ACT] raw=" << currentA[i][k] << " clamped=" << a
                  << " act_val=" << act_val
                  << " bit_round=" << state.current_bit_round << " bit=" << bit
                  << std::endl;
    }
    if (currentA[i][k] == 0.0f || weight == 0.0f) {
        state.current_k++;

        if (state.current_k < currentK) {
            state.current_bit_round = 0;
            state.partial_sum = 0.0f;
            issueBit(idx, pe);
            return;
        } else {
            handleTaskCompletion(idx);
        }
        return;
    }

    if (mulUnits[pe]->isFull()) {
        stallMulQueue.push({bit_slice, weight, idx});
        if (!retryEvent.scheduled()) {
            schedule(retryEvent, curTick() + 1);
        }
    } else {
        mulUnits[pe]->push(bit_slice, weight, idx);
    }
}

void
PadeDriver::onProductReady(int pe, float result, int idx)
{
    auto &state = padeTracker[idx];
    int b = state.current_bit_round;

    float significance;
    if (b == 0) {
        significance = -128.0f;
    } else {
        significance = std::pow(2.0f, 7 - b);
    }

    state.partial_sum += result * significance;

    state.current_bit_round++;
    if (idx < 5) {
        std::cout << "[PRODUCT] idx=" << idx << " result=" << result
                  << " significance=" << significance
                  << " partial_sum=" << state.partial_sum << std::endl;
    }

    if (state.current_bit_round == PADE_GUARD_BIT) {
        std::cout << "[PRUNE_CHECK] idx=" << idx
                  << " upper_bound=" << (state.partial_sum)
                  << " rowMax=" << rowMax[idx / currentCols]
                  << " decision=" << checkBUI_GF(idx) << std::endl;
    }

    if (state.current_bit_round == PADE_GUARD_BIT && checkBUI_GF(idx)) {
        state.is_pruned = true;
        prunedTokens++;
        handleTaskCompletion(idx);
        return;
    }

    if (state.current_bit_round < TOTAL_BITS) {
        if (state.is_pruned) {
            return;
        }
        issueBit(idx, pe);
    } else {
        if (!accUnits[pe]->push(state.partial_sum, idx)) {
            stallAccQueue.push({state.partial_sum, idx});
        }
    }
}

bool
PadeDriver::checkBUI_GF(int idx)
{
    auto &state = padeTracker[idx];
    float max_remaining =
        (currentK - state.current_k) * rowMax[idx / currentCols];

    float bit_uncertainty = (std::pow(2, 8 - state.current_bit_round) - 1);

    float upper_bound = state.partial_sum + max_remaining * bit_uncertainty;
    return false;
    return upper_bound < (rowMax[idx / currentCols] * 1.0f);
}

void
PadeDriver::handleTaskCompletion(int idx)
{
    auto &state = padeTracker[idx];
    state.total_dot_prod /= (127.0f);
    int row = idx / currentCols;
    int col = idx % currentCols;
    if (idx < 5) {
        float naive = 0;
        int row = idx / currentCols;
        int col = idx % currentCols;

        for (int kk = 0; kk < currentK; kk++) {
            float w = isTranspose ? currentB[col][kk] : currentB[kk][col];
            naive += currentA[row][kk] * w;
        }

        std::cout << "[GROUND_TRUTH] idx=" << idx << " naive=" << naive
                  << " approx=" << state.total_dot_prod << std::endl;
    }
    currentOut[row][col] = state.total_dot_prod;

    completedTasks++;
    if (idx < 5) {
        std::cout << "[FINAL] idx=" << idx << " value=" << state.total_dot_prod
                  << std::endl;
    }
    activeTasks.erase(std::remove(activeTasks.begin(), activeTasks.end(), idx),
                      activeTasks.end());
    tryDispatchNext();

    if (completedTasks == totalTasks) {
        if (phase == PHASE_PROJ) {
            onProjectionDone();
        } else if (phase == PHASE_QK) {
            onQKDone();
        } else if (phase == PHASE_AV) {
            onAVDone();
        }
    }
}

void
PadeDriver::regStats()
{
    using namespace statistics;
    SimObject::regStats();
    numReads.name(name() + ".numReads")
        .desc("Number of reads performed by the baseline driver");
    numWrites.name(name() + ".numWrites")
        .desc("Number of writes performed by the baseline driver");
    stallCycles.name(name() + ".stallCycles")
        .desc("Number of cycles stalled due to full PE queues");
}
} // namespace gem5

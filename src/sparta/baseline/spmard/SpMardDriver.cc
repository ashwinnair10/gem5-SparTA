#include "sparta/baseline/spmard/SpMardDriver.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5
{

SpMardDriver::SpMardDriver(const SpMardDriverParams &p)
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
}

void
SpMardDriver::tick()
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
SpMardDriver::startup()
{
    std::cout << "[SPMARD] startup with " << numPEs << " PEs\n";
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
SpMardDriver::start()
{
    std::cout << "[SPMARD] Starting projection (SPMARD)\n";
    phase = PHASE_PROJ;
    startProjection();
}

void
SpMardDriver::startProjection()
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

    dispatchMatMul(X, weight, out, M, Kdim, Kdim);
}

void
SpMardDriver::onProjectionDone()
{
    projPart++;
    std::cout << "[SPMARD] Projection part " << projPart << " done.\n";

    if (projPart < 3) {
        startProjection();
        return;
    }

    std::cout << "[SPMARD] Projection done. Starting QK\n";

    phase = PHASE_QK;
    currentOut = Scores;
    currentCols = N;

    numReads += M * Kdim;
    numReads += N * Kdim;
    numWrites += M * N;
    dispatchMatMul(Q, K, Scores, M, N, Kdim, true);
}

void
SpMardDriver::dispatchMatMul(float **A, float **B, float **C, int M, int N,
                             int K, bool transpose)
{
    partialSums.assign(M * N, 0.0f);
    remainingCounts.assign(M * N, 0);
    totalTasks = M * N;
    completedTasks = 0;
    currentDataflow = chooseDataflow(A, B, M, N, K, transpose);
    std::cout << "Dataflow : " << currentDataflow << std::endl;

    switch (currentDataflow) {
        case IP_M:
            dispatchIP(A, B, M, N, K, transpose, true);
            break;
        case IP_N:
            dispatchIP(A, B, M, N, K, transpose, false);
            break;

        case OP_M:
            dispatchOP(A, B, M, N, K, transpose, true);
            break;
        case OP_N:
            dispatchOP(A, B, M, N, K, transpose, false);
            break;

        case ROW_M:
            dispatchROW(A, B, M, N, K, transpose, true);
            break;
        case ROW_N:
            dispatchROW(A, B, M, N, K, transpose, false);
            break;
    }

    for (int idx = 0; idx < M * N; idx++) {
        if (remainingCounts[idx] == 0) {
            currentOut[idx / currentCols][idx % currentCols] = 0.0f;
            completedTasks++;
        }
        if (completedTasks == totalTasks) {
            if (phase == PHASE_PROJ) {
                onProjectionDone();
            } else if (phase == PHASE_QK) {
                onQKDone();
            } else if (phase == PHASE_AV) {
                onAVDone();
            }
            return;
        }
    }

    std::cout << "[SPMARD] Sparse dispatch done\n";
}

void
SpMardDriver::dispatchIP(float **A, float **B, int M, int N, int K,
                         bool transpose, bool A_stationary)
{
    buildCSR(A, M, K, A_csr);

    if (!transpose) {
        buildCSR(B, K, N, B_csr);
    } else {
        buildCSRTranspose(B, N, K, B_csr);
    }

    if (A_stationary) {
        for (int i = 0; i < M; i++) {
            for (int a_idx = A_csr.row_ptr[i]; a_idx < A_csr.row_ptr[i + 1];
                 a_idx++) {

                float a = A_csr.values[a_idx];
                int k = A_csr.col_idx[a_idx];

                for (int b_idx = B_csr.row_ptr[k];
                     b_idx < B_csr.row_ptr[k + 1]; b_idx++) {

                    float b = B_csr.values[b_idx];
                    int j = B_csr.col_idx[b_idx];

                    int idx = i * N + j;
                    int pe = idx % numPEs;

                    remainingCounts[idx]++;

                    if (!mulUnits[pe]->push(a, b, idx)) {
                        stallMulQueue.push({a, b, idx});
                        if (!retryEvent.scheduled()) {
                            schedule(retryEvent, curTick() + 1);
                        }
                    }
                }
            }
        }
    } else {
        for (int k = 0; k < K; k++) {
            for (int b_idx = B_csr.row_ptr[k]; b_idx < B_csr.row_ptr[k + 1];
                 b_idx++) {

                float b = B_csr.values[b_idx];
                int j = B_csr.col_idx[b_idx];

                for (int i = 0; i < M; i++) {
                    float a = A[i][k];
                    if (a == 0.0f) {
                        continue;
                    }

                    int idx = i * N + j;
                    int pe = idx % numPEs;

                    remainingCounts[idx]++;

                    if (!mulUnits[pe]->push(a, b, idx)) {
                        stallMulQueue.push({a, b, idx});
                        if (!retryEvent.scheduled()) {
                            schedule(retryEvent, curTick() + 1);
                        }
                    }
                }
            }
        }
    }
}
void
SpMardDriver::dispatchOP(float **A, float **B, int M, int N, int K,
                         bool transpose, bool A_stationary)
{
    std::cout << "DISPATCH OP" << std::endl;
    for (int k = 0; k < K; k++) {

        for (int i = 0; i < M; i++) {
            float a = A[i][k];
            if (a == 0.0f) {
                continue;
            }

            for (int j = 0; j < N; j++) {

                float b = transpose ? B[j][k] : B[k][j];
                if (b == 0.0f) {
                    continue;
                }

                int idx = i * N + j;
                int pe = A_stationary ? (k % numPEs) : (j % numPEs);
                remainingCounts[idx]++;

                if (!mulUnits[pe]->push(a, b, idx)) {
                    stallMulQueue.push({a, b, idx});
                    if (!retryEvent.scheduled()) {
                        schedule(retryEvent, curTick() + 1);
                    }
                }
            }
        }
    }
}

void
SpMardDriver::dispatchROW(float **A, float **B, int M, int N, int K,
                          bool transpose, bool A_stationary)
{
    std::cout << "DISPATCH ROW" << std::endl;
    for (int i = 0; i < M; i++) {

        for (int k = 0; k < K; k++) {
            float a = A[i][k];
            if (a == 0.0f) {
                continue;
            }

            for (int j = 0; j < N; j++) {

                float b = transpose ? B[j][k] : B[k][j];
                if (b == 0.0f) {
                    continue;
                }

                int idx = i * N + j;
                int pe = A_stationary ? (i % numPEs) : (j % numPEs);

                remainingCounts[idx]++;

                if (!mulUnits[pe]->push(a, b, idx)) {
                    stallMulQueue.push({a, b, idx});
                    if (!retryEvent.scheduled()) {
                        schedule(retryEvent, curTick() + 1);
                    }
                }
            }
        }
    }
}

SpMardDriver::Dataflow
SpMardDriver::chooseDataflow(float **A, float **B, int M, int N, int K,
                             bool transpose)
{
    int nnz_MK = countNNZ(A, M, K);
    int nnz_KN;
    if (!transpose) {
        nnz_KN = countNNZ(B, K, N);
    } else {
        nnz_KN = countNNZ(B, N, K);
    }

    if (K <= numPEs) {
        return (nnz_KN > nnz_MK) ? IP_N : IP_M;
    } else if (M * N > numPEs * mulUnits[0]->queueSize()) {
        return (nnz_KN > nnz_MK) ? ROW_N : ROW_M;
    } else {
        return (nnz_KN > nnz_MK) ? OP_N : OP_M;
    }
}

void
SpMardDriver::onProductReady(int pe, float product, int idx)
{
    if (!accUnits[pe]->push(product, idx)) {
        stallAccQueue.push(std::make_pair(product, idx));
        if (!retryEvent.scheduled()) {
            schedule(retryEvent, curTick() + 1);
        }
        return;
    }
}

void
SpMardDriver::onAccReady(int pe, float sum, int idx)
{
    partialSums[idx] += sum;
    remainingCounts[idx]--;
    // std::cerr << "[DEBUG] idx=" << idx << " remaining=" <<
    // remainingCounts[idx]
    //           << std::endl;
    if (remainingCounts[idx] == 0) {
        currentOut[idx / currentCols][idx % currentCols] = partialSums[idx];
        completedTasks++;
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
}

void
SpMardDriver::retryStalled()
{
    bool stalled = false;

    size_t mul_sz = stallMulQueue.size();
    for (size_t i = 0; i < mul_sz; i++) {
        auto [a, b, idx] = stallMulQueue.front();
        stallMulQueue.pop();

        int mul_id = idx % numPEs;
        if (mulUnits[mul_id]->push(a, b, idx)) {
        } else {
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
SpMardDriver::onQKDone()
{
    std::cout << "[SPMARD] QK done. Running softmax.\n";
    phase = PHASE_SOFTMAX;
    runSoftmax();
    startAV();
}

void
SpMardDriver::runSoftmax()
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
SpMardDriver::startAV()
{
    phase = PHASE_AV;
    std::cout << "[SPMARD] Starting A*V\n";
    currentOut = Output;
    currentCols = Kdim;
    numReads += M * N;
    numReads += N * Kdim;
    numWrites += M * Kdim;

    dispatchMatMul(Prob, V, Output, M, Kdim, N);
}

void
SpMardDriver::onAVDone()
{
    std::cout << "[SPMARD] AV done. Baseline complete.\n";
    phase = PHASE_DONE;
    exitSimLoop("baseline done");
}

void
SpMardDriver::buildCSR(float **mat, int rows, int cols, CSR &csr)
{
    csr.values.clear();
    csr.col_idx.clear();
    csr.row_ptr.clear();

    csr.row_ptr.resize(rows + 1);
    csr.row_ptr[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = mat[i][j];

            if (val != 0.0f) {
                csr.values.push_back(val);
                csr.col_idx.push_back(j);
            }
        }

        csr.row_ptr[i + 1] = csr.values.size();
    }
}

void
SpMardDriver::buildCSRTranspose(float **mat, int rows, int cols, CSR &csr)
{
    csr.values.clear();
    csr.col_idx.clear();
    csr.row_ptr.assign(cols + 1, 0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mat[i][j] != 0) {
                csr.row_ptr[j + 1]++;
            }
        }
    }

    for (int i = 1; i <= cols; i++) {
        csr.row_ptr[i] += csr.row_ptr[i - 1];
    }

    csr.values.resize(csr.row_ptr[cols]);
    csr.col_idx.resize(csr.row_ptr[cols]);

    std::vector<int> offset = csr.row_ptr;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mat[i][j] != 0) {
                int pos = offset[j]++;
                csr.values[pos] = mat[i][j];
                csr.col_idx[pos] = i;
            }
        }
    }
}

int
SpMardDriver::countNNZ(float **mat, int rows, int cols)
{
    int nnz = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mat[i][j] != 0.0f) {
                nnz++;
            }
        }
    }
    return nnz;
}

void
SpMardDriver::regStats()
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

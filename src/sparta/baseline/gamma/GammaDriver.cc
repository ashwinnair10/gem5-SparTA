#include "sparta/baseline/gamma/GammaDriver.hh"

#include <cmath>
#include <iostream>

#include "sim/sim_exit.hh"
#include "sparta/PE_Acc.hh"
#include "sparta/PE_Mul.hh"

namespace gem5
{

GammaDriver::GammaDriver(const GammaDriverParams &p)
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
      phase(PHASE_IDLE),

      startEvent([this] { start(); }, "gamma_start_event"),
      retryEvent([this] { retryStalled(); }, "gamma_retry_event"),
      tickEvent([this] { tick(); }, "gamma_tick_event")

{
    transpose = false;
    softmaxRow = new float[N];
    currentOut = nullptr;

    pe_k.resize(numPEs, 0);
    pe_j.resize(numPEs, 0);

    for (auto *m : p.mul_units) {
        mulUnits.push_back(m);
    }

    for (auto *a : p.acc_units) {
        accUnits.push_back(a);
    }

    localAcc.resize(numPEs);
    rowAssigned.resize(numPEs, -1);
    remainingOps.resize(numPEs, 0);
    issueDone.resize(numPEs, false);
}

void
GammaDriver::startup()
{
    std::cout << "[GAMMA] startup with " << numPEs << " PEs\n";

    for (int i = 0; i < numPEs; i++) {
        mulUnits[i]->setCallback([this, i](float product, int col) {
            this->onProductReady(i, product, col);
        });

        accUnits[i]->setCallback(
            [this, i](float sum, int col) { this->onAccReady(i, sum, col); });
    }

    schedule(startEvent, curTick() + 1);
    schedule(tickEvent, curTick() + 1);
}

void
GammaDriver::tick()
{
    if (phase == PHASE_DONE) {
        return;
    }

    if (!stallMulQueue.empty() || !stallAccQueue.empty()) {
        stallCycles++;
    }

    schedule(tickEvent, curTick() + 1);
}

void
GammaDriver::start()
{
    std::cout << "[GAMMA] Starting projection\n";
    phase = PHASE_PROJ;
    startProjection();
}

void
GammaDriver::startProjection()
{
    float **out;
    float **weight;

    std::cout << "[GAMMA] Starting projection part " << projPart << "\n";

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

    currentA = X;
    currentB = weight;
    currentOut = out;
    currentCols = Kdim;
    currentKDim = Kdim;
    numReads += M * Kdim;
    numReads += Kdim * Kdim;
    numWrites += M * Kdim;

    dispatchMatMul(M);
}

void
GammaDriver::onProjectionDone()
{
    projPart++;

    if (projPart < 3) {
        startProjection();
        return;
    }

    std::cout << "[GAMMA] Projection done. Starting QK\n";

    phase = PHASE_QK;

    currentA = Q;
    currentB = K;
    currentOut = Scores;
    currentCols = N;
    currentKDim = Kdim;
    transpose = true;
    numReads += M * Kdim;
    numReads += N * Kdim;
    numWrites += M * N;

    dispatchMatMul(M);
}

void
GammaDriver::onQKDone()
{
    transpose = false;
    std::cout << "[GAMMA] QK done. Running softmax\n";

    phase = PHASE_SOFTMAX;

    runSoftmax();

    startAV();
}

void
GammaDriver::runSoftmax()
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
GammaDriver::startAV()
{
    std::cout << "[GAMMA] Starting AV\n";

    phase = PHASE_AV;

    currentA = Prob;
    currentB = V;
    currentOut = Output;
    currentCols = Kdim;
    currentKDim = N;
    transpose = false;
    numReads += M * N;
    numReads += N * Kdim;
    numWrites += M * Kdim;

    dispatchMatMul(M);
}

void
GammaDriver::onAVDone()
{
    std::cout << "[GAMMA] done\n";
    phase = PHASE_DONE;
    exitSimLoop("gamma done");
}

void
GammaDriver::dispatchMatMul(int rows)
{
    while (!rowQueue.empty()) {
        rowQueue.pop();
    }

    for (int i = 0; i < rows; i++) {
        rowQueue.push(i);
    }

    for (int pe = 0; pe < numPEs; pe++) {
        rowAssigned[pe] = -1;
        remainingOps[pe] = 0;
        localAcc[pe].clear();
    }

    A_csr.values.clear();
    A_csr.col_idx.clear();
    A_csr.row_ptr.clear();

    B_csr.values.clear();
    B_csr.col_idx.clear();
    B_csr.row_ptr.clear();

    buildCSR(currentA, rows, currentKDim, A_csr);

    if (!transpose) {
        buildCSR(currentB, currentKDim, currentCols, B_csr);
    } else {
        buildCSRTranspose(currentB, currentCols, currentKDim, B_csr);
    }

    assignRows();
}

void
GammaDriver::assignRows()
{
    for (int pe = 0; pe < numPEs; pe++) {
        if (rowAssigned[pe] != -1) {
            continue;
        }

        if (rowQueue.empty()) {
            return;
        }

        int row = rowQueue.front();
        rowQueue.pop();

        rowAssigned[pe] = row;
        issueDone[pe] = false;

        localAcc[pe].clear();
        remainingOps[pe] = 0;

        pe_k[pe] = 0;
        pe_j[pe] = 0;
        issueRow(pe, row);
    }
}

void
GammaDriver::issueRow(int pe, int row)
{
    int a_start = A_csr.row_ptr[row];
    int a_end = A_csr.row_ptr[row + 1];

    while (pe_k[pe] < (a_end - a_start)) {
        int a_idx = a_start + pe_k[pe];

        float a = A_csr.values[a_idx];
        int k = A_csr.col_idx[a_idx];

        int b_start = B_csr.row_ptr[k];
        int b_end = B_csr.row_ptr[k + 1];

        while (pe_j[pe] < (b_end - b_start)) {
            int b_idx = b_start + pe_j[pe];

            float b = B_csr.values[b_idx];
            int j = B_csr.col_idx[b_idx];

            if (!mulUnits[pe]->push(a, b, j)) {
                // std::cout << RED << "[FAILED]" << RESET << " PE " << pe
                //           << " stalled on mul for row " << row << " k=" << k
                //           << " j=" << j << "\n";
                stallMulQueue.push({pe, row, pe_k[pe], pe_j[pe]});
                // std::cout << "STallMUlQueue size: " << stallMulQueue.size()
                //           << "\n";

                if (!retryEvent.scheduled()) {
                    schedule(retryEvent, curTick() + 1);
                }

                return;
            }

            remainingOps[pe]++;
            pe_j[pe]++;
        }

        pe_j[pe] = 0;
        pe_k[pe]++;
    }

    issueDone[pe] = true;
    if (remainingOps[pe] == 0) {
        finishRow(pe);
    }
}

void
GammaDriver::onProductReady(int pe, float product, int col)
{
    if (!accUnits[pe]->push(product, col)) {
        // std::cout << YELLOW << "[FAILED]" << RESET << " PE " << pe
        //           << " stalled on acc for product " << product
        //           << " col=" << col << "\n";
        stallAccQueue.push({pe, product, col});
        // std::cout << "STallAccQueue size: " << stallAccQueue.size() << "\n";
        if (!retryEvent.scheduled()) {
            schedule(retryEvent, curTick() + 1);
        }
    }
    // std::cout << GREEN << "[SUCCESS]" << RESET << " PE " << pe
    //           << " issued acc for product " << product << " col=" << col
    //           << "\n";
    if (!retryEvent.scheduled() && !stallMulQueue.empty()) {
        schedule(retryEvent, curTick() + 1);
    }
}

void
GammaDriver::onAccReady(int pe, float sum, int col)
{
    localAcc[pe][col] += sum;

    remainingOps[pe]--;

    if (remainingOps[pe] == 0 && issueDone[pe]) {
        finishRow(pe);
    }
}

void
GammaDriver::finishRow(int pe)
{
    int row = rowAssigned[pe];
    for (int j = 0; j < currentCols; j++) {
        currentOut[row][j] = 0;
    }

    for (auto &e : localAcc[pe]) {
        currentOut[row][e.first] = e.second;
    }

    rowAssigned[pe] = -1;

    assignRows();

    checkComplete();
}

void
GammaDriver::retryStalled()
{
    bool progress = false;

    size_t acc_sz = stallAccQueue.size();
    for (size_t i = 0; i < acc_sz; i++) {
        auto t = stallAccQueue.front();
        stallAccQueue.pop();

        if (accUnits[t.pe]->push(t.product, t.col)) {
            progress = true;
        } else {
            stallAccQueue.push(t);
        }
    }

    size_t mul_sz = stallMulQueue.size();
    for (size_t i = 0; i < mul_sz; i++) {
        auto t = stallMulQueue.front();
        stallMulQueue.pop();

        int pe = t.pe;
        int row = t.row;

        if (rowAssigned[pe] != row) {
            continue;
        }

        pe_k[pe] = t.k;
        pe_j[pe] = t.j;

        issueRow(pe, row);

        progress = true;
    }

    if (progress && (!stallMulQueue.empty() || !stallAccQueue.empty()) &&
        !retryEvent.scheduled()) {
        schedule(retryEvent, curTick() + 1);
    }
}

void
GammaDriver::checkComplete()
{
    if (!rowQueue.empty()) {
        return;
    }

    for (int pe = 0; pe < numPEs; pe++) {
        if (rowAssigned[pe] != -1) {
            return;
        }
    }

    std::cout << "[GAMMA] Matmul complete for phase " << phase << "\n";
    // std::cout << "[GAMMA] Stall Queue\n";
    // std::cout << "[GAMMA] Mul Queue Size: " << stallMulQueue.size() << "\n";
    // std::cout << "[GAMMA] Acc Queue Size: " << stallAccQueue.size() << "\n";

    if (phase == PHASE_PROJ) {
        onProjectionDone();
    } else if (phase == PHASE_QK) {
        onQKDone();
    } else if (phase == PHASE_AV) {
        onAVDone();
    }
}

void
GammaDriver::regStats()
{
    SimObject::regStats();

    stallCycles.name(name() + ".stallCycles")
        .desc("Number of cycles where the driver was stalled");
    numReads.name(name() + ".numReads")
        .desc("Number of reads issued to the PEs");
    numWrites.name(name() + ".numWrites")
        .desc("Number of writes issued to the PEs");
}

void
GammaDriver::buildCSR(float **mat, int rows, int cols, CSR &csr)
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
GammaDriver::buildCSRTranspose(float **mat, int rows, int cols, CSR &csr)
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

} // namespace gem5

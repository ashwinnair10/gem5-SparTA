#include "sparta/MatMul.hh"

#include <iostream>

#include "sim/sim_exit.hh"

namespace gem5{
    MatMul::MatMul(const MatMulParams &p)
        : SimObject(p),
        mul(p.mul),
        acc(p.acc),
        retryEvent([this]{ retryStalled(); },
                    "sparta_matmul_retry_event"),
        tickEvent([this]{tick(); },
                  "sparta_matmul_tick_event")
        {
            transpose=false;
        }

    void MatMul::tick(){
        bool hasPending =
        !stallMulQueue.empty() ||
        !stallAccQueue.empty();

        if (hasPending)
            stallCycles++;
        schedule(tickEvent, curTick() + 1);
    }
    void MatMul::startup(){
        std::cout << "[SparTA-MATMUL] startup\n";
        schedule(tickEvent, curTick()+1);
    }
    void MatMul::startMatMul(uint64_t A_ptr,
                         uint64_t B_ptr,
                         uint64_t C_ptr,
                         int _M, int _N, int _K,bool t)
    {

        A = reinterpret_cast<float **>(A_ptr);
        B = reinterpret_cast<float **>(B_ptr);
        C = reinterpret_cast<float **>(C_ptr);
        M = _M;
        N = _N;
        K = _K;
        transpose = t;

        partialSums.assign(M * N, 0.0f);
        remainingCounts.assign(M * N, K);


        mul->setCallback([&](float product,int idx) {
            this->onMulDone(product,idx);
        });
        acc->setCallback([&](float sum,int idx) {
            this->onAccDone(sum,idx);
        });
        done=false;
        i = j = k = 0;
        if (!mul->push(A[i][k],transpose ? B[j][k] : B[k][j],i*N+j)){
            stallMulQueue.push({A[i][k], transpose ? B[j][k] : B[k][j],i*N+j});
            if (!retryEvent.scheduled())
                schedule(retryEvent, curTick() + 1);
        }
    }

    void MatMul::onMulDone(float product,int idx){
        if (!acc->push(product,idx)){
            stallAccQueue.push({product,idx});
            if (!retryEvent.scheduled())
                schedule(retryEvent, curTick() + 1);
            return;
        }
    }

    void MatMul::onAccDone(float partial,int idx){

        partialSums[idx] += partial;
        remainingCounts[idx]--;
        if (remainingCounts[idx]>0){
            k++;

            if (!mul->push(A[i][k],transpose
                ? B[j][k] : B[k][j],i*N+j)){
                stallMulQueue.push({A[i][k],
                     transpose ? B[j][k] : B[k][j],i*N+j});
                if (!retryEvent.scheduled())
                    schedule(retryEvent, curTick() + 1);
            }
            return;
        }
        C[i][j]=partialSums[idx];
        j++;
        if (j<N){
            k=0;

            if (!mul->push(A[i][k],transpose
                ? B[j][k] : B[k][j],i*N+j)){
                stallMulQueue.push({A[i][k],
                    transpose ? B[j][k] : B[k][j],i*N+j});
                if (!retryEvent.scheduled())
                    schedule(retryEvent, curTick() + 1);
            }
            return;
        }
        i++;
        if (i<M){
            j=0;
            k=0;

            if (!mul->push(A[i][k],transpose
                ? B[j][k] : B[k][j],i*N+j)){
                stallMulQueue.push({A[i][k],
                    transpose ? B[j][k] : B[k][j],i*N+j});
                if (!retryEvent.scheduled())
                    schedule(retryEvent, curTick() + 1);
            }
            return;
        }
        std::cout << "[SparTA-MatMul] Finished matrix multiplication @ tick "
                    << curTick() << "\n";
        if (finishedCallback){
            finishedCallback();
        }
        done=true;
    }

    void MatMul::retryStalled(){

        bool stalled = false;

        size_t msz = stallMulQueue.size();
        for (size_t i = 0; i < msz; i++) {
            auto [a, b, idx] = stallMulQueue.front();
            stallMulQueue.pop();

            if (!mul->push(a, b, idx)) {
                stallMulQueue.push({a, b, idx});
                stalled = true;
            }
        }

        size_t asz = stallAccQueue.size();
        for (size_t i = 0; i < asz; i++) {
            auto [prod, idx] = stallAccQueue.front();
            stallAccQueue.pop();

            if (!acc->push(prod, idx)) {
                stallAccQueue.push({prod, idx});
                stalled = true;
            }
        }

        if (stalled || !stallMulQueue.empty() || !stallAccQueue.empty()) {
            schedule(retryEvent, curTick() + 1);
        }

    }

    void MatMul::regStats(){
        using namespace statistics;
        stallCycles
            .name(name() + ".stall_cycles")
            .desc("Number of cycles the MatMul unit was stalled")
            .flags(nozero);
    }
}

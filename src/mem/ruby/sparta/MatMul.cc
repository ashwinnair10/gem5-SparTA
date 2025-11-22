#include "mem/ruby/sparta/MatMul.hh"

#include <iostream>

#include "sim/sim_exit.hh"

namespace gem5{
    MatMul::MatMul(const MatMulParams &p)
        : SimObject(p),
        mul(p.mul),
        acc(p.acc),
        A(reinterpret_cast<float **>(p.A)),
        B(reinterpret_cast<float **>(p.B)),
        C(reinterpret_cast<float **>(p.C)),
        M(p.M),
        N(p.N),
        K(p.K){}
    void MatMul::startup(){
        std::cout << "[SparTA-MATMUL] startup\n";
    }
    void MatMul::startMatMul(uint64_t A_ptr,
                         uint64_t B_ptr,
                         uint64_t C_ptr,
                         int _M, int _N, int _K)
    {
        A = reinterpret_cast<float **>(A_ptr);
        B = reinterpret_cast<float **>(B_ptr);
        C = reinterpret_cast<float **>(C_ptr);
        M = _M;
        N = _N;
        K = _K;

        mul->callback = [&](float product) {
            this->onMulDone(product);
        };
        acc->callback = [&](float sum, int remaining) {
            this->onAccDone(sum, remaining);
        };

        i = j = k = 0;
        acc->reset(K);
        mul->startCompute(A[i][k], B[k][j]);
    }

    void MatMul::onMulDone(float product){
        acc->feedProduct(product);
    }

    void MatMul::onAccDone(float partial,int remaining_ops){
        if (remaining_ops>0){
            k++;
            mul->startCompute(A[i][k],B[k][j]);
            return;
        }
        C[i][j]=partial;
        j++;
        if (j<N){
            k=0;
            acc->reset(K);
            mul->startCompute(A[i][k],B[k][j]);
            return;
        }
        i++;
        if (i<M){
            j=0;
            k=0;
            acc->reset(K);
            mul->startCompute(A[i][k],B[k][j]);
            return;
        }
        std::cout << "[SparTA-MatMul] Finished matrix multiplication @ tick "
                    << curTick() << "\n";
        exitSimLoop("");
    }
}

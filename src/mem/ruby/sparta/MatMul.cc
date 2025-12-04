#include "mem/ruby/sparta/MatMul.hh"

#include <iostream>

#include "sim/sim_exit.hh"

namespace gem5{
    MatMul::MatMul(const MatMulParams &p)
        : SimObject(p),
        mul(p.mul),
        acc(p.acc)
        {}
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

        mul->setCallback([&](float product,int idx) {
            this->onMulDone(product,idx);
        });
        acc->setCallback([&](float sum, int remaining,int idx) {
            this->onAccDone(sum, remaining,idx);
        });
        done=false;
        i = j = k = 0;
        acc->reset(K);
        mul->startCompute(A[i][k], B[k][j],i*N+j);
    }

    void MatMul::onMulDone(float product,int idx){
        acc->feedProduct(product,idx);
    }

    void MatMul::onAccDone(float partial,int remaining_ops,int idx){
        if (remaining_ops>0){
            k++;
            mul->startCompute(A[i][k],B[k][j],i*N+j);
            return;
        }
        C[i][j]=partial;
        j++;
        if (j<N){
            k=0;
            acc->reset(K);
            mul->startCompute(A[i][k],B[k][j],i*N+j);
            return;
        }
        i++;
        if (i<M){
            j=0;
            k=0;
            acc->reset(K);
            mul->startCompute(A[i][k],B[k][j],i*N+j);
            return;
        }
        std::cout << "[SparTA-MatMul] Finished matrix multiplication @ tick "
                    << curTick() << "\n";
        if(finishedCallback){
            finishedCallback();
        }
        done=true;
    }
}

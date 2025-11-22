#include "mem/ruby/sparta/PE_Acc.hh"

#include <iostream>

namespace gem5{
    PE_Acc::PE_Acc(const PE_AccParams &p)
        : SimObject(p),
        latency(p.latency),
        computeEvent([this]{ finishCompute(); },
                    "sparta_acc_compute_event"),
        current_sum(0.0f),
        current_input(0.0f),
        remaining_ops(0)
    { }

    void PE_Acc::startup()
    {
        std::cout << "[SparTA-ACC] startup. Latency = "
                << latency << " cycles\n";
    }

    void PE_Acc::reset(int num_ops){
        current_sum=0.0f;
        remaining_ops=num_ops;
        std::cout << "[SparTA-ACC] Reset\n";
    }

    void PE_Acc::feedProduct(float product){
        if (remaining_ops==0)return;
        current_input=product;
        schedule(computeEvent,curTick()+latency);
    }

    void PE_Acc::finishCompute()
    {
        current_sum+=current_input;
        remaining_ops--;
        std::cout << "[SparTA-ACC] Accumulated Partial Sum : "
                << current_sum << "\n";
        if (callback){
            callback(current_sum,remaining_ops);
        }
    }
}

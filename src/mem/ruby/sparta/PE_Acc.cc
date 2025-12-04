#include "mem/ruby/sparta/PE_Acc.hh"

#include <iostream>

namespace gem5{
    PE_Acc::PE_Acc(const PE_AccParams &p)
        : SimObject(p),
        latency(p.latency),
        island(p.island),
        computeEvent([this]{ finishCompute(); },
                    "sparta_acc_compute_event"),
        current_sum(0.0f),
        current_input(0.0f),
        remaining_ops(0)
    { }

    void PE_Acc::startup()
    {
        std::cout << "[SparTA-ACC-"<<island<<"] startup. Latency = "
                << latency << " cycles\n";
    }

    void PE_Acc::reset(int num_ops){
        current_sum=0.0f;
        remaining_ops=num_ops;
        std::cout << "[SparTA-ACC-"<<island<<"] Reset\n";
    }

    void PE_Acc::processNext(){
        if (!inputQueue.empty()){
            current_input=inputQueue.front().first;
            id=inputQueue.front().second;
            inputQueue.pop();
            schedule(computeEvent,curTick()+latency);
        }
    }

    void PE_Acc::feedProduct(float product,int i){
        inputQueue.push({product,i});
        if (!computeEvent.scheduled()){
            processNext();
        }
    }

    void PE_Acc::finishCompute()
    {
        current_sum+=current_input;
        remaining_ops--;
        std::cout << "[SparTA-ACC-"<<island<<"] Accumulated Partial Sum : "
                << current_sum << "\n";
        if (callback){
            callback(current_sum,remaining_ops,id);
        }
        processNext();
    }
}

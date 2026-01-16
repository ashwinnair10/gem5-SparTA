#include "sparta/PE_Acc.hh"

#include <iostream>

namespace gem5{
    PE_Acc::PE_Acc(const PE_AccParams &p)
        : SimObject(p),
        latency(p.latency),
        island(p.island),
        queue_size(p.queue_size),
        computeEvent([this]{ finishCompute(); },
                    "sparta_acc_compute_event"),
        current_sum(0.0f),
        current_input(0.0f),
        remaining_ops(0)
    {
        busy=false;
     }

    void PE_Acc::startup()
    {
        std::cout << YELLOW<< "[SparTA-ACC-"<<island<<"] startup. Latency = "
                << latency << " cycles\n"<< RESET;
    }

    void PE_Acc::reset(int num_ops){
        current_sum=0.0f;
        remaining_ops=num_ops;
        while (!inputQueue.empty()) inputQueue.pop();
        busy=false;
        std::cout << YELLOW<< "[SparTA-ACC-"<<island<<"] Reset\n"<< RESET;
    }

    void PE_Acc::processNext(){
        if (computeEvent.scheduled())
            return;
        if (busy) return;
        if (inputQueue.empty()||remaining_ops<=0){
            idleCycles++;
            return;
        }
        busy=true;
        current_input=inputQueue.front().first;
        id=inputQueue.front().second;
        inputQueue.pop();
        std::cout << YELLOW
            << "[SparTA-ACC-"<<island<<"] Start -- PartialSum : "
            << current_sum << " , Input : " << current_input
            << " -- index: " << id
            << " --  @ tick " << curTick() << "\n"<< RESET;
        schedule(computeEvent,curTick()+latency);
    }

    void PE_Acc::setParams(float sum, int remaining){
        current_sum=sum;
        remaining_ops=remaining;
    }

    // void PE_Acc::feedProduct(float product,int i){
    //     inputQueue.push({product,i});
    //     if (!computeEvent.scheduled()){
    //         processNext();
    //     }
    // }

    bool PE_Acc::push(float product,int i)
    {
        if (inputQueue.size()==queue_size){
            return false;
        }
        inputQueue.push({product,i});
        if (!computeEvent.scheduled()){
            processNext();
        }
        return true;
    }

    //acc has to be modified to push out the partial sum
    // instead of keeping it till remaining_ops is 0?
    void PE_Acc::finishCompute()
    {
        busy=false;
        current_sum+=current_input;
        remaining_ops--;
        numAccOps++;
        activeCycles+=latency;
        std::cout << YELLOW
            << "[SparTA-ACC-"<<island<<"] Accumulated Partial Sum : "
            << current_sum << "\n"<< RESET;
        if (callback){
            callback(current_sum,remaining_ops,id);
        }
        if (remaining_ops>0&&!computeEvent.scheduled())
        processNext();
    }

    void PE_Acc::regStats()
    {
        using namespace statistics;
        numAccOps
            .name(name() + ".numAccOps")
            .desc("Number of accumulate operations performed")
            .prereq(numAccOps)
            ;
        activeCycles
            .name(name() + ".activeCycles")
            .desc("Number of cycles the Accumulator was active")
            .prereq(activeCycles)
            ;
        idleCycles
            .name(name() + ".idleCycles")
            .desc("Number of cycles the Accumulator was idle")
            .prereq(idleCycles)
            ;
    }
}

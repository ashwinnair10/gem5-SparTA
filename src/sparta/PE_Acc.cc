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
        tickEvent([this]{ tick(); },
                  "sparta_acc_tick_event"),
        current_sum(0.0f),
        current_input(0.0f),
        remaining_ops(0)
    {
        busy=false;
    }

    void PE_Acc::tick()
    {
        if (busy){
            activeCycles++;
        } else {
            idleCycles++;
        }
        schedule(tickEvent, curTick() + 1);
    }

    void PE_Acc::startup()
    {
        // std::cout << YELLOW<< "[SparTA-ACC-"<<island<<"] startup.
        // Latency = "
        //         << latency << " cycles\n"<< RESET;
        schedule(tickEvent, curTick()+1);
    }

    void PE_Acc::processNext(){
        if (computeEvent.scheduled())
            return;
        if (busy) return;
        if (inputQueue.empty()){
            return;
        }
        busy=true;
        current_input=inputQueue.front().first;
        id=inputQueue.front().second;
        inputQueue.pop();
        // std::cout << YELLOW
        //     << "[SparTA-ACC-"<<island<<"] Start -- PartialSum : "
        //     << current_sum << " , Input : " << current_input
        //     << " -- index: " << id
        //     << " --  @ tick " << curTick() << "\n"<< RESET;
        schedule(computeEvent,curTick()+latency);
    }

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

    void PE_Acc::finishCompute()
    {
        numAccOps++;
        busy=false;
        // std::cout << YELLOW
        //     << "[SparTA-ACC-"<<island<<"] Accumulated Partial Sum : "
        //     << current_sum << "\n"<< RESET;
        if (callback){
            callback(current_input,id);
        }
        if (!computeEvent.scheduled())
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

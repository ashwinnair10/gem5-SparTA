#include "sparta/PE_Mul.hh"

#include <iostream>

namespace gem5{
    PE_Mul::PE_Mul(const PE_MulParams &p)
        : SimObject(p),
        latency(p.latency),
        island(p.island),
        computeEvent([this]{ finishCompute(); },
                    "sparta_mul_compute_event")
    {
    }

    void PE_Mul::startup()
    {
        std::cout << GREEN<< "[SparTA-MUL-"<<island<<"] startup. Latency = "
                << latency << " cycles\n"<< RESET;
    }

    void PE_Mul::processNext(){
        if (!inputQueue.empty()){
            std::tuple<float,float,int> operands=inputQueue.front();
            inputQueue.pop();
            operand1=std::get<0>(operands);
            operand2=std::get<1>(operands);
            id=std::get<2>(operands);
            result=0.0f;
            std::cout << GREEN
                << "[SparTA-MUL-"<<island<<"] Start -- Operand 1 : "
                << operand1 << " , Operand 2 : " << operand2
                << " -- index: " << std::get<2>(operands)
                << " --  @ tick " << curTick() << "\n"<< RESET;
            schedule(computeEvent, curTick() + latency);
        }
    }

    void PE_Mul::startCompute(float val1,float val2,int i)
    {
        inputQueue.push({val1,val2,i});
        if (!computeEvent.scheduled()){
            processNext();
        }
    }

    void PE_Mul::finishCompute()
    {
        result=operand1*operand2;
        std::cout << GREEN
            << "[SparTA-MUL-"<<island<<"] Finished compute -- Result : "
            << result << " -- @ tick " << curTick() << "\n"<< RESET;
        if (callback){
            callback(result,id);
        }
        processNext();
    }
}

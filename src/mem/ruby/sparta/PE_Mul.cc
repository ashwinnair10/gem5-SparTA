#include "mem/ruby/sparta/PE_Mul.hh"

#include <iostream>

namespace gem5{
    PE_Mul::PE_Mul(const PE_MulParams &p)
        : SimObject(p),
        latency(p.latency),
        computeEvent([this]{ finishCompute(); },
                    "sparta_mul_compute_event")
    { }

    void PE_Mul::startup()
    {
        std::cout << "[SparTA-MUL] startup. Latency = "
                << latency << " cycles\n";
    }

    void PE_Mul::startCompute(float val1,float val2)
    {
        operand1=val1;
        operand2=val2;
        result=0.0f;
        std::cout << "[SparTA-MUL] Start -- Operand 1 : "
                << operand1 << " , Operand 2 : " << operand2
                << " --  @ tick " << curTick() << "\n";
        schedule(computeEvent, curTick() + latency);
    }

    void PE_Mul::finishCompute()
    {
        result=operand1*operand2;
        std::cout << "[SparTA-MUL] Finished compute -- Result : "
                << result << " -- @ tick " << curTick() << "\n";
        if (callback){
            callback(result);
        }
    }
}

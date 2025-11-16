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

        //temporary test
        startCompute(0);

    }

    void PE_Mul::startCompute(int block_id)
    {
        std::cout << "[SparTA-MUL] Start block " << block_id
                << " @ tick " << curTick() << "\n";

        schedule(computeEvent, curTick() + latency);
    }

    void PE_Mul::finishCompute()
    {
        std::cout << "[SparTA-MUL] Finished compute @ tick "
                << curTick() << "\n";
    }
}

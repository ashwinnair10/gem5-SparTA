#include "sparta/PE_Mul.hh"

#include <iostream>

namespace gem5
{
PE_Mul::PE_Mul(const PE_MulParams &p)
    : SimObject(p),
      latency(p.latency),
      island(p.island),
      queue_size(p.queue_size),
      computeEvent([this] { finishCompute(); }, "sparta_mul_compute_event"),
      tickEvent([this] { tick(); }, "sparta_mul_tick_event"),
      busy(false)
{}

void
PE_Mul::tick()
{
    if (busy) {
        activeCycles++;
    } else {
        idleCycles++;
    }
    schedule(tickEvent, curTick() + 1);
}

void
PE_Mul::startup()
{
    // std::cout << GREEN<< "[SparTA-MUL-"<<island<<"] startup. Latency = "
    //         << latency << " cycles\n"<< RESET;
    schedule(tickEvent, curTick() + 1);
}

void
PE_Mul::processNext()
{
    if (computeEvent.scheduled()) {
        return;
    }
    if (!inputQueue.empty()) {
        std::tuple<float, float, int> operands = inputQueue.front();
        inputQueue.pop();
        operand1 = std::get<0>(operands);
        operand2 = std::get<1>(operands);
        id = std::get<2>(operands);
        result = 0.0f;
        busy = true;
        // std::cout << GREEN
        //     << "[SparTA-MUL-"<<island<<"] Start -- Operand 1 : "
        //     << operand1 << " , Operand 2 : " << operand2
        //     << " -- index: " << std::get<2>(operands)
        //     << " --  @ tick " << curTick() << "\n"<< RESET;
        schedule(computeEvent, curTick() + latency);
    }
}

bool
PE_Mul::push(float val1, float val2, int i)
{
    if (inputQueue.size() == queue_size) {
        return false;
    }
    inputQueue.push({val1, val2, i});
    if (!computeEvent.scheduled()) {
        processNext();
    }
    return true;
}

void
PE_Mul::finishCompute()
{
    result = operand1 * operand2;
    // std::cout << GREEN
    //     << "[SparTA-MUL-"<<island<<"] Finished compute -- Result : "
    //     << result << " -- @ tick " << curTick() << "\n"<< RESET;
    numMulOps++;
    busy = false;
    if (callback) {
        callback(result, id);
    }
    if (!computeEvent.scheduled()) {
        processNext();
    }
}

void
PE_Mul::regStats()
{
    using namespace statistics;
    SimObject::regStats();
    numMulOps.name(name() + ".numMulOps")
        .desc("Number of multiplication operations performed by PE_Mul");
    activeCycles.name(name() + ".activeCycles")
        .desc("Number of active cycles of PE_Mul");
    idleCycles.name(name() + ".idleCycles")
        .desc("Number of idle cycles of PE_Mul");
}
} // namespace gem5

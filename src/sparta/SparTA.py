from m5.params import *
from m5.SimObject import (
    SimObject,
)


class PE_Mul(SimObject):
    type = "PE_Mul"
    cxx_class = "gem5::PE_Mul"
    cxx_header = "sparta/PE_Mul.hh"
    latency = Param.Int(10, "Compute latency per block")
    island = Param.Int(0, "Island number")
    queue_size = Param.Int(16, "Size of the input queue")


class PE_Acc(SimObject):
    type = "PE_Acc"
    cxx_class = "gem5::PE_Acc"
    cxx_header = "sparta/PE_Acc.hh"
    latency = Param.Int(10, "Compute latency per block")
    island = Param.Int(0, "Island Number")
    queue_size = Param.Int(16, "Size of the input queue")


class MatMul(SimObject):
    type = "MatMul"
    cxx_class = "gem5::MatMul"
    cxx_header = "sparta/MatMul.hh"
    mul = Param.PE_Mul("Multiplier PE")
    acc = Param.PE_Acc("Accumulator PE")


class BaselineDriverSequential(SimObject):
    type = "BaselineDriverSequential"
    cxx_class = "gem5::BaselineDriverSequential"
    cxx_header = "sparta/baseline/sequential/BaselineDriverSequential.hh"

    mm = Param.MatMul("MatMul unit")

    X = Param.Addr("Pointer to input matrix X")
    WQ = Param.Addr("Pointer to weight matrix WQ")
    WK = Param.Addr("Pointer to weight matrix WK")
    WV = Param.Addr("Pointer to weight matrix WV")

    Q = Param.Addr("Pointer to Q matrix")
    K = Param.Addr("Pointer to K^T matrix")
    V = Param.Addr("Pointer to V matrix")
    Scores = Param.Addr("Pointer to Scores output")
    Prob = Param.Addr("Pointer to Probability (softmax) matrix")
    Output = Param.Addr("Pointer to Output (A*V)")

    M = Param.Int("Rows")
    N = Param.Int("Columns")
    Kdim = Param.Int("Depth")


class BaselineDriverParallel(SimObject):
    type = "BaselineDriverParallel"
    cxx_class = "gem5::BaselineDriverParallel"
    cxx_header = "sparta/baseline/parallel/BaselineDriverParallel.hh"

    numPEs = Param.Int(0, "Number of PE_Mul/PE_Acc units")
    mul_units = VectorParam.PE_Mul([], "List of PE_Mul units")
    acc_units = VectorParam.PE_Acc([], "List of PE_Acc units")

    X = Param.Addr("Pointer to input matrix X")
    WQ = Param.Addr("Pointer to weight matrix WQ")
    WK = Param.Addr("Pointer to weight matrix WK")
    WV = Param.Addr("Pointer to weight matrix WV")
    Q = Param.Addr("Pointer to Q")
    K = Param.Addr("Pointer to K")
    V = Param.Addr("Pointer to V")
    Scores = Param.Addr("Pointer to Scores")
    Prob = Param.Addr("Pointer to Probabilities")
    Output = Param.Addr("Pointer to Final Output")
    M = Param.Int("Seq length")
    N = Param.Int("Seq length again")
    Kdim = Param.Int("Head dim")


class AcceleratorDriver(SimObject):
    type = "AcceleratorDriver"
    cxx_class = "gem5::AcceleratorDriver"
    cxx_header = "sparta/accelerator/AcceleratorDriver.hh"

    numPEs = Param.Int(0, "Number of PE_Mul/PE_Acc units")
    mul_units = VectorParam.PE_Mul([], "List of PE_Mul units")
    acc_units = VectorParam.PE_Acc([], "List of PE_Acc units")
    mul_queue_depth = Param.Int("Depth of PE_Mul queue")
    acc_queue_depth = Param.Int("Depth of PE_Acc queue")

    X = Param.Addr("Pointer to input matrix X")
    WQ = Param.Addr("Pointer to weight matrix WQ")
    WK = Param.Addr("Pointer to weight matrix WK")
    WV = Param.Addr("Pointer to weight matrix WV")
    Q = Param.Addr("Pointer to Q")
    K = Param.Addr("Pointer to K")
    V = Param.Addr("Pointer to V")
    Scores = Param.Addr("Pointer to Scores")
    Prob = Param.Addr("Pointer to Probabilities")
    Output = Param.Addr("Pointer to Final Output")
    M = Param.Int("Seq length")
    N = Param.Int("Seq length again")
    Kdim = Param.Int("Head dim")


class GammaDriver(SimObject):
    type = "GammaDriver"
    cxx_header = "sparta/baseline/gamma/GammaDriver.hh"
    cxx_class = "gem5::GammaDriver"

    numPEs = Param.Int(0, "Number of PE_Mul/PE_Acc units")
    mul_units = VectorParam.PE_Mul([], "List of PE_Mul units")
    acc_units = VectorParam.PE_Acc([], "List of PE_Acc units")
    mul_queue_depth = Param.Int(4, "Depth of PE_Mul queue")
    acc_queue_depth = Param.Int(4, "Depth of PE_Acc queue")

    X = Param.Addr("Pointer to input matrix X")
    WQ = Param.Addr("Pointer to weight matrix WQ")
    WK = Param.Addr("Pointer to weight matrix WK")
    WV = Param.Addr("Pointer to weight matrix WV")
    Q = Param.Addr("Pointer to Q")
    K = Param.Addr("Pointer to K")
    V = Param.Addr("Pointer to V")
    Scores = Param.Addr("Pointer to Scores")
    Prob = Param.Addr("Pointer to Probabilities")
    Output = Param.Addr("Pointer to Final Output")
    M = Param.Int("Seq length")
    N = Param.Int("Seq length again")
    Kdim = Param.Int("Head dim")

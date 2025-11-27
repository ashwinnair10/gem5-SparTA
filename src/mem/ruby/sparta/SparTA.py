from m5.params import *
from m5.SimObject import (
    SimObject,
    cxxMethod,
)


class PE_Mul(SimObject):
    type = "PE_Mul"
    cxx_class = "gem5::PE_Mul"
    cxx_header = "mem/ruby/sparta/PE_Mul.hh"
    latency = Param.Int(10, "Compute latency per block")

    @cxxMethod
    def startCompute(self, operand1, operand2):
        pass


class PE_Acc(SimObject):
    type = "PE_Acc"
    cxx_class = "gem5::PE_Acc"
    cxx_header = "mem/ruby/sparta/PE_Acc.hh"
    latency = Param.Int(10, "Compute latency per block")

    @cxxMethod
    def reset(self, num_ops):
        pass

    @cxxMethod
    def feedProduct(self, product):
        pass


class MatMul(SimObject):
    type = "MatMul"
    cxx_class = "gem5::MatMul"
    cxx_header = "mem/ruby/sparta/MatMul.hh"
    mul = Param.PE_Mul("Multiplier PE")
    acc = Param.PE_Acc("Accumulator PE")


class BaselineDriver(SimObject):
    type = "BaselineDriver"
    cxx_class = "gem5::BaselineDriver"
    cxx_header = "mem/ruby/sparta/baseline/sequential/BaselineDriver.hh"

    mm = Param.MatMul("MatMul unit")

    Q = Param.Addr("Pointer to Q matrix")
    K = Param.Addr("Pointer to K^T matrix")
    V = Param.Addr("Pointer to V matrix")
    Scores = Param.Addr("Pointer to Scores output")
    Prob = Param.Addr("Pointer to Probability (softmax) matrix")
    Output = Param.Addr("Pointer to Output (A*V)")

    M = Param.Int("Rows")
    N = Param.Int("Columns")
    Kdim = Param.Int("Depth")

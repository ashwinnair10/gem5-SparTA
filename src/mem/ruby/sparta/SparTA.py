from m5.params import *
from m5.SimObject import SimObject


class PE_Mul(SimObject):
    type = "PE_Mul"
    cxx_class = "gem5::PE_Mul"
    cxx_header = "mem/ruby/sparta/PE_Mul.hh"
    latency = Param.Int(10, "Compute latency per block")

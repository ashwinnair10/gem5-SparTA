from m5.objects import *

system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

# ---- Bus ----
system.membus = SystemXBar()

# ---- Memory ----
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# ---- ASADI ----
system.asadi = ASADIDriver(test_addr=0x100000)
system.asadi.mem_port = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

print("Beginning simulation!")
m5.simulate()

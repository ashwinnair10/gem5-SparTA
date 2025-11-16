from m5.objects import *

root = Root(full_system=False)
root.mul0 = PE_Mul(latency=5)

m5.instantiate()

exit_event = m5.simulate()
print("Exit at tick:", m5.curTick(), exit_event.getCause())

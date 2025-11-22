import m5
from m5.objects import *

root = Root(full_system=False)
root.mul0 = PE_Mul(latency=5)

m5.instantiate()
root.mul0.startCompute(5, 6)

print("Running simulation...")
exit_event = m5.simulate()

print("Exit at tick:", m5.curTick())
print("Cause:", exit_event.getCause())

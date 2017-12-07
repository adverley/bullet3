import pybullet as p
import time

p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.loadSoftBodyFromObj()
p.loadMJCF("ground_plane.xml")

while 1:
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()

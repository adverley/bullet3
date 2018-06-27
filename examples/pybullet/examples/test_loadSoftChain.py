import pybullet as p
import time
import math

p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
pos = [0.0, 0.0, 1.0]
orn = p.getQuaternionFromEuler([math.pi / 4., math.pi / 4., math.pi / 4.])
#orn = p.getQuaternionFromEuler([0, 0, math.pi / 2.])
softUid = p.loadSoftWire(40, 0.3, pos, orn)
print("SoftUid", softUid)
print("SoftUid", p.getBasePositionAndOrientation(softUid))

posOrn = p.getBasePositionAndOrientation(softUid)
angle = p.getEulerFromQuaternion(posOrn[1])
angle = [angle[0], angle[1], angle[2] + math.pi/2.]
print("BasePos and Orientation", posOrn)
newUid = p.loadURDF("torus/torus.urdf", posOrn[0],orn)
p.loadMJCF("ground_plane.xml")

# Spawn block to test soft body properties of wire
#p.loadURDF("block_rot.urdf", [0, -.8, 1], globalScaling=2)

#p.setRealTimeSimulation(1)
#p.setTimeStep(0.005)

while 1:
    time.sleep(0.01)
    p.stepSimulation()

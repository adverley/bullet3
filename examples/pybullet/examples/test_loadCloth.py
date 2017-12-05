import pybullet as p

p.connect(p.GUI)
p.loadCloth()
p.setGravity(0, 0, -9.8)

while 1:
    p.stepSimulation()

p.disconnect()

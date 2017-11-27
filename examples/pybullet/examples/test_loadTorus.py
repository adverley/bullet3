import pybullet as p

p.connect(p.GUI)
p.loadTorus("torus/torus.urdf", [0, .7, .2])

while 1:
    pass

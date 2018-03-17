import pybullet as p
import time
import math
from datetime import datetime

p.connect(p.GUI)
p.loadURDF("plane.urdf", [0,0,-0.3],useFixedBase=True)
baxter_id = p.loadURDF("baxter_common/baxter.urdf",[0,0,0.5],useFixedBase=True)

joint_name2joint_index = {}
for joint_nr in range(p.getNumJoints(baxter_id)):
    joint_info = p.getJointInfo(baxter_id, joint_nr)
    joint_idx = joint_info[0]
    joint_name = joint_info[1]
    joint_name2joint_index[joint_name] = joint_idx
    #print joint_name, ":", joint_idx
    print "motorinfo:", joint_info[3], joint_info[1], joint_info[0]
print "\n\n\n"
print joint_name2joint_index

while True:
    p.stepSimulation()

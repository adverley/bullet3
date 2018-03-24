import pybullet as p
import time
import math
import numpy as np
from datetime import datetime

# lower limits for null space
ll = [-.967, -2	, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2	, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0, 0, 0, 0, 0, 0, 0, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# Function to test the inverse kinematics function on the right arm of the baxter robot
def move_baxter_gripper(baxterId, pos, orn, baxterEndEffectorIndex=26):
    jointPoses = p.calculateInverseKinematics(
        baxterId, baxterEndEffectorIndex, pos, orn, ll, ul, jr, rp)

    for i in range(len(jointPoses)):
        p.setJointMotorControl2(bodyIndex=baxterId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i], targetVelocity=0, force=10000, positionGain=1, velocityGain=1)


# Load in Baxter and print joint information
p.connect(p.GUI)
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
baxterId = p.loadURDF("baxter_common/baxter.urdf",
                      [0, 0, 0.5], useFixedBase=True)
numJoints = p.getNumJoints(baxterId)

joint_name2joint_index = {}
for joint_nr in range(p.getNumJoints(baxterId)):
    joint_info = p.getJointInfo(baxterId, joint_nr)
    joint_idx = joint_info[0]
    joint_name = joint_info[1]
    joint_name2joint_index[joint_name] = joint_idx
    # print joint_name, ":", joint_idx
    print "motorinfo:", joint_info[3], joint_info[1], joint_info[0]
print "\n\n\n"
print joint_name2joint_index
# p.loadSoftBody("wire_softbody.obj")

# Open right arm gripper
"""
p.setJointMotorControl2(
    baxterId, 27, controlMode=p.POSITION_CONTROL, targetPosition=1, force=10000)
p.setJointMotorControl2(
    baxterId, 29, controlMode=p.POSITION_CONTROL, targetPosition=1, force=10000)
for i in range(10):
    p.stepSimulation()
"""

# Load in environment (block, torus) and spawn the block in the gripper of the baxter
block_coord = [0.85, 0.95, 0.425]  # Left hand gripper coordinates
blockId = p.loadURDF("block_rot.urdf", block_coord)
# p.loadSoftBody("wire_softbody.obj")
torusId = p.loadURDF("torus/torus.urdf", [1.3, 0, .7],
                     p.getQuaternionFromEuler([0, 0, math.pi / 2.]))  # [1.8, 0, .7] for torus_only.obj
#orn = p.getQuaternionFromEuler([math.pi / 2., math.pi / 2., math.pi])
#block_coord = [0.87, 0.98, 0.825]
#p.resetBasePositionAndOrientation(blockId, block_coord, orn)

# Place block in the right hand gripper
# block_coord = [0.865, -1.059, 0.825] #Vertical coordinates
# orn = p.getQuaternionFromEuler([math.pi / 2., math.pi / 2., math.pi])
# block_coord = [0.89, -1.07, 0.822]  # Horizontal coordinates with opening gripper
# Horizontal coordinates without opening gripper
block_coord = [0.875, -1.07, 0.822]
orn = p.getQuaternionFromEuler(
    [math.pi, math.pi, 3. * math.pi / 4.])
p.resetBasePositionAndOrientation(blockId, block_coord, orn)

time.sleep(5)

"""
coord1 = p.getLinkState(baxterId, 27)[0]
coord2 = p.getLinkState(baxterId, 29)[0]
block_coord = [(x[0] + x[1]) / 2. for x in zip(coord1, coord2)]
p.resetBasePositionAndOrientation(blockId, block_coord, orn)
"""

# Get global position of joint in world
print "Link state:", p.getLinkState(baxterId, 27)
print "Link state block:", p.getLinkState(blockId, 0)

# Close the right hand gripper
p.setJointMotorControl2(
    baxterId, 27, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)
p.setJointMotorControl2(
    baxterId, 29, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)
p.stepSimulation()

# p.setGravity(0, 0, -9.8)

# Test camera
head_camera_index = 9
t_v = [.95, 0, -.1]  # Translation of the camera position
cam_pos = np.array(p.getLinkState(baxterId, head_camera_index)[0]) + t_v
print cam_pos
p.resetDebugVisualizerCamera(1.3, 180, -41, cam_pos)

look = cam_pos
distance = 1.
pitch = -20  # -10
yaw = -100  # 245
roll = 0
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    look, distance, yaw, pitch, roll, 2)
fov = 85.
aspect = 640. / 480.
near = 0.01
far = 10
proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

img_arr = p.getCameraImage(width=640, height=480,
                           viewMatrix=view_matrix,
                           projectionMatrix=proj_matrix)

orn = p.getQuaternionFromEuler([0, -math.pi, 0])

print("Start simulation")
t = 0.
trailDuration = 15

# Spawn a second block to test getConactPoints with torus
b2_coord = p.getBasePositionAndOrientation(torusId)[0]
print "Torus center:", b2_coord
block2Id = p.loadURDF("block.urdf", b2_coord)  # [1.13, 0, .95]

torus_pos = p.getBasePositionAndOrientation(torusId)[0]
block2_pos = p.getBasePositionAndOrientation(block2Id)[0]

x_bool = torus_pos[0] <= block2_pos[0]
y_bool = torus_pos[1] - \
    0.3 < block2_pos[1] and torus_pos[1] + 0.3 > block2_pos[1]
z_bool = torus_pos[2] - \
    0.3 < block2_pos[2] and torus_pos[2] + 0.3 > block2_pos[2]
if x_bool and y_bool and z_bool:
    print "Block within the hole. block_pos:", block2_pos, "torus_pos:", torus_pos
else:
    print "Block not within the hole. block_pos:", block2_pos, "torus_pos:", torus_pos

# Create invisible plane within torus and calculate contact points with that?

print "Contact point(s) torus and block2: ", p.getContactPoints(torusId, block2Id)

# Baxter open gripper
# set_kuka_gripper(kukaId, 1)
orn = p.getQuaternionFromEuler([0, -math.pi, 0])
# move_baxter_gripper(baxterId, [-.6, 0, .5], orn)
# move_baxter_gripper(baxterId, [-.5, -.025, 0], orn)

# Move the gripper to the block
# set_kuka_gripper(baxterId, 0)
step = 0

while 1:
    p.stepSimulation()

while 0:
    p.stepSimulation()
    step -= 0.05
    p.setJointMotorControl2(
        baxterId, 13, controlMode=p.POSITION_CONTROL, targetPosition=step, force=10000)

    p.setJointMotorControl2(
        baxterId, 27, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)
    p.setJointMotorControl2(
        baxterId, 29, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)

    torus_pos = p.getBasePositionAndOrientation(torusId)[0]
    block2_pos = p.getBasePositionAndOrientation(block2Id)[0]

    x_bool = torus_pos[0] <= block2_pos[0]
    y_bool = torus_pos[1] - \
        0.3 < block2_pos[1] and torus_pos[1] + 0.3 > block2_pos[1]
    z_bool = torus_pos[2] - \
        0.3 < block2_pos[2] and torus_pos[2] + 0.3 > block2_pos[1]

    if x_bool and y_bool and z_bool:
        # print "Block within the hole. block_pos:", block2_pos, "torus_pos:", torus_pos
        pass
    else:
        # print "Block not within the hole. block_pos:", block2_pos, "torus_pos:", torus_pos
        pass

    # Update cam image
    """
    img_arr = p.getCameraImage(width=640, height=480,
                               viewMatrix=view_matrix,
                               projectionMatrix=proj_matrix)
    """
    time.sleep(.001)

    # print p.getContactPoints(baxterId, blockId)

while 0:
    time.sleep(0.02)
    t += 1

    # Kuka open gripper
    # set_kuka_gripper(baxterId, .5)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])
    move_baxter_gripper(baxterId, [1, 1, 1], orn)

    if t > 30:
        move_baxter_gripper(baxterId, [1.5, 1.5, 1.5], orn)

    if t > 180:
        # Test move baxter right arm
        p.setJointMotorControl2(
            baxterId, 15, controlMode=p.POSITION_CONTROL, targetPosition=1.036, force=10000)

        # Test baxter open right arm gripper
        p.setJointMotorControl2(
            baxterId, 27, controlMode=p.POSITION_CONTROL, targetPosition=1, force=10000)
        p.setJointMotorControl2(
            baxterId, 29, controlMode=p.POSITION_CONTROL, targetPosition=1, force=10000)

    p.stepSimulation()

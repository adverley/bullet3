import pybullet as p
import time
import math
from datetime import datetime


def set_kuka_gripper(kukaId, pos, mode=p.POSITION_CONTROL):
    # Varying the position between 0 and 1 opens and closes the gripper for
    # the right base joint
    right_base_joint = 11
    left_base_joint = 8
    fingerJoints = [8, 11]  # 10, 13

    for i in fingerJoints:
        p.setJointMotorControl2(
            kukaId, jointIndex=i, controlMode=mode,
            targetPosition=pos, force=500, positionGain=0.2, velocityGain=1)


def move_kuka_gripper(kukaId, pos, orn, kukaEndEffectorIndex=6):
    jointPoses = p.calculateInverseKinematics(
        kukaId, kukaEndEffectorIndex, pos, orn, ll, ul, jr, rp)

    for i in range(numJoints - 3):
        p.setJointMotorControl2(bodyIndex=kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i], targetVelocity=0, force=500, positionGain=0.2, velocityGain=1)


def move_kuka_param(kukaId, pos, orn, var, maxVal=0., minVal=0., kukaEndEffectorIndex=6):
    """This function will move the Kuka arm to the desired location using a parameter equation.
    The variable var indicates in which direction the movement is varying(x, y, z)."""

    t = minVal
    numJoints = p.getNumJoints(kukaId)
    maxVal = pos[var]

    if var == 0:
        param = [t, pos[1], pos[2]]
    elif var == 1:
        param = [pos[0], t, pos[2]]
    else:
        param = [pos[0], pos[1], t]

    while t < maxVal:
        jointPoses = p.calculateInverseKinematics(
            kukaId, kukaEndEffectorIndex, pos, orn, ll, ul, jr, rp)

        for i in range(numJoints - 6):
            p.setJointMotorControl2(bodyIndex=kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=500, positionGain=0.2, velocityGain=1)

        t = t + timeStep


clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
    p.connect(p.GUI)
p.loadURDF("plane.urdf", [0, 0, -0.3])
midPoint = [0, .7, .2]
# kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
kukaId = p.loadSDF("kuka_iiwa/kuka_with_gripper.sdf")
kukaId = kukaId[0]
torusID = p.loadURDF("torus/torus.urdf", midPoint)

# p.loadURDF("table/table.urdf", 0.5000000, 0.00000, -.820000, 0.000000,
# 0.000000, 0.0, 1.0)
p.loadURDF("tray/tray.urdf", [-.6, 0, -0.3])
p.loadURDF("block_rot.urdf", [-.5, 0, -0.2])
# p.loadBullet("wire_softbody.bullet")

p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints < 7):
    exit()

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

# Print joint info
joint_name2joint_index = {}
for joint_nr in range(p.getNumJoints(kukaId)):
    joint_info = p.getJointInfo(kukaId, joint_nr)
    joint_idx = joint_info[0]
    joint_name = joint_info[1]
    joint_name2joint_index[joint_name] = joint_idx
print(joint_name2joint_index)

# Reset joint state
for i in range(numJoints):
    p.resetJointState(kukaId, i, rp[i])

useOrientation = 0  # used to be 1 but what does this do???
# If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
# This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)
flag = 0

orn = p.getQuaternionFromEuler([0, -math.pi, 0])
positions = [{"pos": [-.6, 0, .5], "state": 0, "var": 0}, {"pos": [-.5, -.025, 0], "state": 0, "var": 2}, {"pos": [0, 0, .2], "state": 0, "var": 2},
             {"pos": [0, 2, .2], "state": 0, "var": 1}]  # Coordinate to move to + open or close the gripper
maxNumSteps = 10000
timeStep = 0.01
sleepTime = 0.01

print("Start simulation")
beforeTime = time.time()
p.setTimeStep(timeStep)

# set the gravity acceleration
#p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0
trailDuration = 15

# while 1:
#    if (useRealTimeSimulation):
#        dt = datetime.now()
#        t = (dt.second / 60.) * 2. * math.pi
#    else:
#pos = positions[i]["pos"]
#gripper_status = positions[i]["state"]

#move_kuka_param(kukaId, pos, orn, positions[i]["var"])
#set_kuka_gripper(kukaId, gripper_status)

# Kuka open gripper
set_kuka_gripper(kukaId, 1)
orn = p.getQuaternionFromEuler([0, -math.pi, 0])
move_kuka_gripper(kukaId, [-.6, 0, .5], orn)
move_kuka_gripper(kukaId, [-.5, -.025, 0], orn)

# Move the gripper to the block
set_kuka_gripper(kukaId, 0)

while 1:
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
    else:
        #t = t + 0.03
        time.sleep(0.02)
        t += 1

        # Kuka open gripper
        set_kuka_gripper(kukaId, .5)
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        move_kuka_gripper(kukaId, [-.6, 0, .5], orn)

        if t > 30:
            move_kuka_gripper(kukaId, [-.45, -.015, -.05], orn)

        # Move the gripper to the block
        if t > 80:
            set_kuka_gripper(kukaId, 0)

        if t > 110:
            move_kuka_gripper(kukaId, [-.5, -.015, .2], orn)

        if t > 130:
            move_kuka_gripper(kukaId, [0, .4, .5], orn)

        if t > 160:
            move_kuka_gripper(kukaId, [0, 2, .5], orn)

    if (useSimulation and useRealTimeSimulation == 0):
        p.stepSimulation()

    # for i in range(1):
    #     pos = [0, t, .5]
    #     # end effector points down, not up (in case useOrientation==1)
    #     orn = p.getQuaternionFromEuler([0, -math.pi, 0])
    #
    #     jointPoses = p.calculateInverseKinematics(
    #         kukaId, kukaEndEffectorIndex, pos, orn, jointDamping=jd)
    #
    #     if (useSimulation):
    #         for i in range(numJoints - 6):
    #             p.setJointMotorControl2(bodyIndex=kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
    #                                     targetPosition=jointPoses[i], targetVelocity=0, force=500, positionGain=0.2, velocityGain=1)
    #     else:
    #         # reset the joint state (ignoring all dynamics, not recommended to
    #         # use during simulation)
    #         for i in range(numJoints):  # -2 is necesarry for it to work with the gripper
    #             print(p.getJointInfo(kukaId, i))
    #             p.resetJointState(kukaId, i, jointPoses[i])
    #
    # ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    # if (hasPrevPose):
    #     p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    #     p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    # prevPose = pos
    # prevPose1 = ls[4]
    # hasPrevPose = 1

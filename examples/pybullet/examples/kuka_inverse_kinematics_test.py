import pybullet as p
import time
import math
from datetime import datetime


def set_kuka_gripper(pos, jointID=11, mode=p.POSITION_CONTROL):
    # Varying the position between 0 and 1 opens and closes the gripper for
    # the right base joint
    right_base_joint = 11
    left_base_joint = 8
    fingerJoints = [10, 13]

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


clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
    p.connect(p.GUI)
p.loadURDF("plane.urdf", [0, 0, -0.3])
midPoint = [0, 1, .2]
#kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
kukaId = p.loadSDF("kuka_iiwa/kuka_with_gripper.sdf")
kukaId = kukaId[0]
torusID = p.loadURDF("torus/torus.urdf", midPoint)

#p.loadURDF("table/table.urdf", 0.5000000, 0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)
p.loadURDF("tray/tray.urdf", [-.6, 0, -0.3])
p.loadURDF("block_rot.urdf", [-.6, 0, -0.2])

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

# set the gravity acceleration
#p.setGravity(0, 0, -9.8)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

useOrientation = 0  # used to be 1 but what does this do???
# If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
# This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)
# trailDuration is duration (in seconds) after debug lines will be removed automatically
# use 0 for no-removal
trailDuration = 15
flag = 0

while 1:
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
    else:
        #t = t + 0.03

        # Kuka open gripper
        set_kuka_gripper(0.5)
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        move_kuka_gripper(kukaId, [-.6, 0, .5], orn)
        move_kuka_gripper(kukaId, [-.5, -.025, 0], orn)

        # Move the gripper to the block
        set_kuka_gripper(-1)

        # if t > 3:
        #    t = 0.

    if (useSimulation and useRealTimeSimulation == 0):
        p.stepSimulation()

    for i in range(1):
        pos = [0, t, .5]
        # end effector points down, not up (in case useOrientation==1)
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        if (useNullSpace == 1):
            if (useOrientation == 1):
                jointPoses = p.calculateInverseKinematics(
                    kukaId, kukaEndEffectorIndex, pos, orn, ll, ul, jr, rp)
            else:
                jointPoses = p.calculateInverseKinematics(
                    kukaId, kukaEndEffectorIndex, pos, lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp)
        else:
            if (useOrientation == 1):
                jointPoses = p.calculateInverseKinematics(
                    kukaId, kukaEndEffectorIndex, pos, orn, jointDamping=jd)
            else:
                jointPoses = p.calculateInverseKinematics(
                    kukaId, kukaEndEffectorIndex, pos)

        if (useSimulation):
            for i in range(numJoints - 3):
                #print("-------- ", len(jointPoses), " ------------------", i, "--------------")
                p.setJointMotorControl2(bodyIndex=kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i], targetVelocity=0, force=500, positionGain=0.2, velocityGain=1)
        else:
            # reset the joint state (ignoring all dynamics, not recommended to
            # use during simulation)
            for i in range(numJoints):  # -2 is necesarry for it to work with the gripper
                print(p.getJointInfo(kukaId, i))
                p.resetJointState(kukaId, i, jointPoses[i])

    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1

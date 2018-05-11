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

def move_baxter_gripper(baxterId, pos, orn=None, baxterEndEffectorIndex=26):
    motorIndices = [1, 2, 3, 4, 5, 6, 7]
    motors = [12, 13, 14, 15, 16, 18, 19]
    jointPoses = p.calculateInverseKinematics(baxterId, baxterEndEffectorIndex, pos)
    jointPoses = [jointPoses[i] for i in motorIndices]
    assert len(jointPoses) == len(motors)

    p.setJointMotorControlArray(
        baxterId,
        motors,
        controlMode=p.POSITION_CONTROL,
        targetPositions=jointPoses
    )

def calculateEndEffectorPos(baxterId, action, baxterEndEffectorIndex=26):
    motorIndices = [1, 2, 3, 4, 5, 6, 7]
    motors = [12, 13, 14, 15, 16, 18, 19]
    old_pos = []

    for i in range(len(motorIndices)):
        joint_state = p.getJointState(baxterId, motors[i])
        old_pos.append(joint_state[0])

    for i in range(len(motorIndices)):
        p.resetJointState(
                        baxterId,
                        motors[i],
                        action[i])

    endEffectorPos = p.getLinkState(baxterId, baxterEndEffectorIndex)[0]

    for i in range(len(motorIndices)):
        p.resetJointState(
                        baxterId,
                        motors[i],
                        old_pos[i])

    return endEffectorPos


def printJointInfo(Uid):
    joint_name2joint_index = {}
    for joint_nr in range(p.getNumJoints(Uid)):
        joint_info = p.getJointInfo(Uid, joint_nr)
        joint_idx = joint_info[0]
        joint_name = joint_info[1]
        joint_name2joint_index[joint_name] = joint_idx
        # print joint_name, ":", joint_idx
        print("motorinfo:", joint_info[3], joint_info[1], joint_info[0])
    print("\n\n\n")
    print(joint_name2joint_index)


def main():
    # Load in Baxter and print joint information
    p.connect(p.GUI)
    p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
    baxterId = p.loadURDF("baxter_common/baxter.urdf",
                          [0, 0, 0.5], useFixedBase=True)
    numJoints = p.getNumJoints(baxterId)
    baxterEndEffectorIndex = 26

    printJointInfo(baxterId)

    block_coord = [0.85, 0.95, 0.425]
    blockId = p.loadURDF("block_rot.urdf", block_coord, globalScaling=0.1)
    torusId = p.loadURDF("torus/torus.urdf", [1.3, 0, .7], p.getQuaternionFromEuler([0, 0, math.pi / 2.]))
    p.createConstraint(baxterId, 26, blockId, -1, jointType=p.JOINT_FIXED, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])

    #Set start position
    for i in range(numJoints):
        p.resetJointState(baxterId, i, 0.)

    # Check location of end effector
    pos = np.array(p.getLinkState(baxterId, 26)[0])
    p.loadURDF("cube.urdf", pos, globalScaling=0.04, useFixedBase=True)

    # Test end effector position calculation
    print("Current endeffectorPos:", p.getLinkState(baxterId, 26)[0])
    print("Update to endeffectorPos:", [0.5, 0.5, 0.5])
    pos = pos + np.array([0.5, 0.5, 0.5])

    jointPoses = p.calculateInverseKinematics(baxterId, baxterEndEffectorIndex, pos)
    jointPoses = [jointPoses[i] for i in motorIndices]
    print("New endeffectorPos:", calculateEndEffectorPos(baxterId, jointPoses))

    print("Start simulation")
    while 1:
        pos = pos + np.array([0, 0.005, 0.005])
        move_baxter_gripper(baxterId, pos)
        #calculateEndEffectorPos(baxterId, jointPoses)
        p.stepSimulation()

if __name__ == '__main__':
    main()

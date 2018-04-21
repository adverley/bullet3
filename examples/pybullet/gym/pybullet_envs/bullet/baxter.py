import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Baxter:
    "This class will only use the right arm of the Baxter robot"

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = pybullet_data.getDataPath()
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 5000.
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.baxterEndEffectorIndex = 19  # or 25
        self.baxterGripperIndex = 20  # or 26
        self.baxterHeadCameraIndex = 9
        # TODO This will have to be changed based on torus URDF scaling factor
        self.torusRad = 0.4
        self.margin = 0.1
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        self.reset()

    def reset(self):
        self.baxterUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "baxter_common/baxter.urdf"), [0, 0, 0.62], useFixedBase=True)
        # p.resetBasePositionAndOrientation(
        #    self.baxterUid, [0, 0, 0.5], [0.000000, 0.000000, 0.000000, 1.000000])
        self.numJoints = p.getNumJoints(self.baxterUid)  # 42

        p.loadURDF(os.path.join(
            self.urdfRootPath, "plane.urdf"), [0, 0, -0.3], useFixedBase=True)

        # Load in torus
        # torus_coord = [1.1, 0, .5]
        orn = p.getQuaternionFromEuler([0, 0, math.pi / 2.])

        ypos = -.1 + 0.05 * np.random.random()
        zpos = .6 + 0.05 * np.random.random()
        torus_coord = [1.2, ypos, zpos]
        # ang = 3.1415925438 * random.random() --> TODO maybe randomize angle in the future as dom randomization
        # orn = p.getQuaternionFromEuler([0, 0, ang])

        self.torusUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "torus/torus.urdf"), torus_coord,
            orn, useFixedBase=True)

        # Compute coordinates of block
        coord1 = p.getLinkState(self.baxterUid, 27)[0]
        coord2 = p.getLinkState(self.baxterUid, 29)[0]
        block_coord = [(x[0] + x[1]) / 2. for x in zip(coord1, coord2)]

        block_coord = [0.875, -1.07, 0.942]  # Horizontal coordinates
        orn = p.getQuaternionFromEuler([math.pi, math.pi, 3. * math.pi / 4.])

        self.blockUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "block_rot.urdf"), block_coord)

        p.resetBasePositionAndOrientation(self.blockUid, block_coord, orn)

        self.motorNames = []
        self.motorIndices = [12, 13, 14, 15, 16,
                             18, 19]  # Right arm joints (27,29) right_gripper left and right finger joints
        # Finger joints are removed since they should not be opened

        for i in self.motorIndices:
            jointInfo = p.getJointInfo(self.baxterUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print "motorname", jointInfo[1]
                self.motorNames.append(str(jointInfo[1]))

    def getActionDimension(self):
        return len(self.motorIndices)

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        # Extend this to give information from all joints in self.motorIndices
        observation = []
        state = p.getLinkState(self.baxterUid, self.baxterGripperIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def printJointInfo(self):
        joint_name2joint_index = {}
        for joint_nr in range(p.getNumJoints(self.baxterUid)):
            joint_info = p.getJointInfo(self.baxterUid, joint_nr)
            # print joint_info[0], joint_info[1]
            joint_idx = joint_info[0]
            joint_name = joint_info[1]
            joint_name2joint_index[joint_name] = joint_idx
            print("motorinfo:", joint_info[3], joint_info[1], joint_info[0])
        print(joint_name2joint_index)

    def applyAction(self, motorCommands):
        for action in range(len(motorCommands)):
            motor = self.motorIndices[action]
            p.setJointMotorControl2(self.baxterUid, motor, controlMode=p.POSITION_CONTROL,
                                    targetPosition=motorCommands[action], force=self.maxForce)

        # Close right arm gripper
        p.setJointMotorControl2(
            self.baxterUid, 27, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)
        p.setJointMotorControl2(
            self.baxterUid, 29, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)

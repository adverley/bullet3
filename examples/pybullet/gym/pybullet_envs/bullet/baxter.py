import os
import inspect
import time
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import math
import pybullet_data

class Baxter:
    """
    This class will only use the right arm of the Baxter robot
    """

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, useBlock=True):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 5000.
        self.baxterEndEffectorIndex = 26  # or 25
        self.baxterGripperIndex = 20  # or 26
        self.baxterHeadCameraIndex = 9
        self.useBlock = useBlock
        self.torusScale = 1.
        self.torusRad = 0.23 * self.torusScale
        self.margin = 0.06
        self.maxIter = 40
        self.llSpace = [0.3, -0.3, 0.9] #x,y,z
        self.ulSpace = [0.9, 0.2, 1.3] #x,y,z
        self.setup()

    def setup(self):
        self.baxterUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "baxter_common/baxter.urdf"), [0, 0, 0.62], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.numJoints = p.getNumJoints(self.baxterUid)  # 42

        p.loadURDF(os.path.join(
            self.urdfRootPath, "plane.urdf"), [0, 0, -0.3], useFixedBase=True)

        # Load in torus
        orn = p.getQuaternionFromEuler([0, 0, math.pi / 2.])

        ypos = -.1 + 0.05 * np.random.random()
        zpos = 1. + 0.05 * np.random.random()
        torus_coord = [1.1, ypos, zpos]
        # ang = 3.1415925438 * random.random() --> TODO maybe randomize angle as dom randomization
        # orn = p.getQuaternionFromEuler([0, 0, ang])

        self.torusUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "torus/torus.urdf"), torus_coord,
            orn, useFixedBase=True, globalScaling=self.torusScale)

        if self.useBlock:
            # Compute coordinates of block
            coord1 = p.getLinkState(self.baxterUid, 28)[0]
            coord2 = p.getLinkState(self.baxterUid, 30)[0]
            block_coord = [(x[0] + x[1]) / 2. for x in zip(coord1, coord2)]
            orn = p.getLinkState(self.baxterUid, 23)[1]  # Get orn from link 23
            # Joint 30 (orthagonal), joint 25 (orthagonal)

            self.blockUid = p.loadURDF(os.path.join(
                self.urdfRootPath, "block_rot.urdf"), block_coord)

            p.resetBasePositionAndOrientation(self.blockUid, block_coord, orn)
            self.cid_base = p.createConstraint(self.baxterUid, self.baxterEndEffectorIndex, self.blockUid, -1,
                                               jointType=p.JOINT_FIXED, jointAxis=[1, 0, 0],
                                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
                                               parentFrameOrientation=p.getQuaternionFromEuler([0, math.pi / 2., 0]))

        # Create line for reward function
        orn = np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.torusUid)[1]))
        orn[2] = orn[2] + math.pi / 2.  # Rotate within plane perpendicular to torus
        line_coord = torus_coord
        # This needs to be made more general if torus orn changes so multiply by dir vector from reward function
        #dir = vdot(dir_vector, np.array([0, 0, -0.2]))
        #line_coord += dir
        line_coord[0] -= 0.2
        self.torusLineUid = p.loadURDF(os.path.join(self.urdfRootPath, "block_line.urdf"),
                                       torus_coord, p.getQuaternionFromEuler(orn), useFixedBase=True)

        self.motorNames = []
        self.motorIndices = [12, 13, 14, 15, 16, 18, 19]

        for i in self.motorIndices:
            jointInfo = p.getJointInfo(self.baxterUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print "motorname", jointInfo[1]
                self.motorNames.append(str(jointInfo[1]))

    def reset(self):
        # reset joint state of baxter
        base_pos = [0 for x in range(len(self.motorIndices))]
        for i in range(len(self.motorIndices)):
            p.resetJointState(self.baxterUid, self.motorIndices[i], base_pos[i])

        # re-randomize torus position
        ypos = -.1 + 0.05 * np.random.random()
        zpos = 1. + 0.05 * np.random.random()
        torus_coord = np.array([1.1, ypos, zpos])
        orn = p.getQuaternionFromEuler([0, 0, math.pi / 2.])
        p.resetBasePositionAndOrientation(self.torusUid, torus_coord, orn)

        # Reset torusLine
        orn = np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.torusUid)[1]))
        orn[2] = orn[2] + math.pi / 2.  # Rotate within plane perpendicular to torus
        line_coord = torus_coord
        p.resetBasePositionAndOrientation(self.torusLineUid, torus_coord, p.getQuaternionFromEuler(orn))

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

    def randomizeGripperPos(self):
        # Randomize the right arm start position
        gripperPos = np.array([np.random.uniform(self.llSpace[i], self.ulSpace[i]) for i in range(len(self.llSpace))])

        iter = 0
        # Loop until arm is in correct position alternatively use for loop with range(50)
        while np.any(np.abs(gripperPos - np.array(self.getEndEffectorPos())) > np.array([0.01, 0.01, 0.01])) and (iter < self.maxIter):
            jointPoses = np.array(self.calculateInverseKinematics(gripperPos))
            iter += 1

            for i in range(len(self.motorIndices)):
                p.resetJointState(self.baxterUid, self.motorIndices[i], jointPoses[i])

        #print("Gripper expected: ", gripperPos, "\t Actual gripper pos:", self.getEndEffectorPos())
        #TODO check collision with torus, if true, recalculate gripperPos

    def calculateInverseKinematics(self, pos, orn=None, ll=None, ul=None, jr=None, rp=None):
        # Use inverse kinematics on the right arm of the baxter robot

        if (ll is None) or (ul is None) or (jr is None) or (rp is None):
            if orn is None:
                jointPoses = p.calculateInverseKinematics(
                    self.baxterUid, self.baxterEndEffectorIndex, pos)
            else:
                jointPoses = p.calculateInverseKinematics(
                    self.baxterUid, self.baxterEndEffectorIndex, pos, orn)
        else:
            if orn is None:
                jointPoses = p.calculateInverseKinematics(
                                self.baxterUid,
                                self.baxterEndEffectorIndex,
                                pos,
                                lowerLimits=ll,
                                upperLimits=ul,
                                jointRanges=jr,
                                restPoses=rp
                                )
            else:
                jointPoses = p.calculateInverseKinematics(
                                self.baxterUid,
                                self.baxterEndEffectorIndex,
                                pos,
                                orn,
                                lowerLimits=ll,
                                upperLimits=ul,
                                jointRanges=jr,
                                restPoses=rp
                                )

        # process action for applyAction
        joints = [1, 2, 3, 4, 5, 6, 7]
        action = [jointPoses[i] for i in joints]
        return action

    def getEndEffectorPos(self):
        return p.getLinkState(self.baxterUid, self.baxterEndEffectorIndex)[0]

    def calculateEndEffectorPos(self, action):
        old_pos = []

        for i in range(len(self.motorIndices)):
            joint_state = p.getJointState(
                            self.baxterUid,
                            self.motorIndices[i]
                        )
            old_pos.append(joint_state[0])

        for i in range(len(self.motorIndices)):
            p.resetJointState(
                            self.baxterUid,
                            self.motorIndices[i],
                            action[i])

        endEffectorPos = p.getLinkState(self.baxterUid, self.baxterEndEffectorIndex)[0]

        for i in range(len(self.motorIndices)):
            p.resetJointState(
                            self.baxterUid,
                            self.motorIndices[i],
                            old_pos[i])

        return endEffectorPos

    def applyAction(self, motorCommands):
        assert len(motorCommands) == len(self.motorIndices)

        p.setJointMotorControlArray(
            self.baxterUid,
            self.motorIndices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=motorCommands
        )

        # Close right arm gripper
        p.setJointMotorControl2(
            self.baxterUid, 27, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)
        p.setJointMotorControl2(
            self.baxterUid, 29, controlMode=p.POSITION_CONTROL, targetPosition=0, force=10000)


    def applyVelocity(self, velocityCommands):
        assert len(velocityCommands) == len(self.motorIndices)

        commands = [x for x in velocityCommands] #Scaling
        forces = [500 for x in range(len(velocityCommands))]

        p.setJointMotorControlArray(
            self.baxterUid,
            self.motorIndices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=commands,
            forces=forces
        )

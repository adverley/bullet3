import math
import gym
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from pkg_resources import parse_version
import logging
import sys

from numpy.linalg import norm
from numpy import vdot

class RewardZoo():
    @classmethod
    def create_function(cls, REWARD_TYPE):
        print(REWARD_TYPE)
        if(REWARD_TYPE == 'finite_line_distance'):
            return cls.finite_line_distance_reward
        elif(REWARD_TYPE == 'line_distance'):
            return cls.line_distance_reward
        elif(REWARD_TYPE == 'sparse'):
            return cls.sparse_reward
        elif(REWARD_TYPE == 'torus_distance'):
            return cls.sparse_reward
        else:
            raise Exception('RewardError', 'Reward not found')

    def finite_line_distance_reward(env):
        """Calculates the reward for the episode.
        """

        torus_pos = np.array(
            p.getBasePositionAndOrientation(env._baxter.torusUid)[0])
        env.logger.debug("Reward torus position: %s" % str(torus_pos))

        if env._useBlock:
            block_pos = np.array(
                p.getBasePositionAndOrientation(env._baxter.blockUid)[0])
        else:
            block_pos = np.array(
                p.getLinkState(env._baxter.baxterUid, 26)[0])  # 26 or avg between 28 and 30

        if env._useBlock:
            distance = p.getClosestPoints(env._baxter.blockUid, env._baxter.torusLineUid, 100, -1, -1)[0][8]
        else:
            distance = p.getClosestPoints(env._baxter.baxterUid, env._baxter.torusLineUid, 100, 26, -1)[0][8]

        reward = -distance

        if(distance < env._baxter.torusRad):
            #distance = norm(torus_pos - block_pos)
            distance = math.sqrt((torus_pos[0] - block_pos[0])**2 +
                                 (torus_pos[1] - block_pos[1])**2)
            reward += 4 - distance

        x_bool = (torus_pos[0] + env._baxter.margin) < block_pos[0]
        y_bool = (
            torus_pos[1] - env._baxter.torusRad) < block_pos[1] and (torus_pos[1] + env._baxter.torusRad) > block_pos[1]
        z_bool = (
            torus_pos[2] - env._baxter.torusRad) < block_pos[2] and (block_pos[2] + env._baxter.torusRad) > block_pos[2]

        cp_list = p.getContactPoints(
            env._baxter.baxterUid, env._baxter.torusUid)
        if any(cp_list):
            reward += env._collision_pen

        if y_bool and z_bool and x_bool:
            env.logger.debug(
                "Block within the hole. block_pos: {0} torus_pos: {1}".format(
                    str(block_pos), str(torus_pos)))
            env.logger.debug("env._envStepCounter: %s" %
                              str(env._envStepCounter))
            reward = reward + 10000

        env.logger.debug("Reward: %s \n" % str(reward))
        return reward

    def line_distance_reward(env):
        """Calculates the reward for the episode.
        """

        torus_pos = np.array(
            p.getBasePositionAndOrientation(env._baxter.torusUid)[0])
        env.logger.debug("Reward torus position: %s" % str(torus_pos))

        if env._useBlock:
            block_pos = np.array(
                p.getBasePositionAndOrientation(env._baxter.blockUid)[0])
        else:
            block_pos = np.array(
                p.getLinkState(env._baxter.baxterUid, 26)[0])  # 26 or avg between 28 and 30

        # Move towards the line through the center of the torus
        orn = np.array(p.getEulerFromQuaternion(
            p.getBasePositionAndOrientation(env._baxter.torusUid)[1]))

        # Euler_angle = [yaw, pitch, roll]
        dir_vector = np.array([math.cos(orn[0]) * math.cos(orn[1]),
                               math.sin(orn[0]) * math.cos(orn[1]),
                               math.sin(orn[1])])

        #x0 = p.getJointState(baxterId, 26)[0]
        x0 = torus_pos
        x1 = block_pos

        distance = norm((x0 - x1) - vdot((x0 - x1), dir_vector) * dir_vector)
        reward = -distance

        if(distance < env._baxter.torusRad):
            #distance = norm(torus_pos - block_pos)
            distance = math.sqrt((torus_pos[0] - block_pos[0])**2 +
                                 (torus_pos[1] - block_pos[1])**2)
            reward += 4 - distance

        x_bool = (torus_pos[0] + env._baxter.margin) < block_pos[0]
        y_bool = (
            torus_pos[1] - env._baxter.torusRad) < block_pos[1] and (torus_pos[1] + env._baxter.torusRad) > block_pos[1]
        z_bool = (
            torus_pos[2] - env._baxter.torusRad) < block_pos[2] and (block_pos[2] + env._baxter.torusRad) > block_pos[2]

        cp_list = p.getContactPoints(
            env._baxter.baxterUid, env._baxter.torusUid)
        if any(cp_list):
            reward += env._collision_pen

        if y_bool and z_bool and x_bool:
            env.logger.debug(
                "Block within the hole. block_pos: {0} torus_pos: {1}".format(
                    str(block_pos), str(torus_pos)))
            env.logger.debug("env._envStepCounter: %s" %
                              str(env._envStepCounter))
            reward = reward + 10000

        env.logger.debug("Reward: %s \n" % str(reward))
        return reward

    def sparse_reward(env):
        """Calculates the reward for the episode.
        """

        torus_pos = np.array(
            p.getBasePositionAndOrientation(env._baxter.torusUid)[0])

        if env._useBlock:
            block_pos = np.array(
                p.getBasePositionAndOrientation(env._baxter.blockUid)[0])
        else:
            block_pos = np.array(
                p.getLinkState(env._baxter.baxterUid, 26)[0])  # 26 or avg between 28 and 30

        x_bool = (torus_pos[0] + env._baxter.margin) < block_pos[0]
        y_bool = (
            torus_pos[1] - env._baxter.torusRad) < block_pos[1] and (torus_pos[1] + env._baxter.torusRad) > block_pos[1]
        z_bool = (
            torus_pos[2] - env._baxter.torusRad) < block_pos[2] and (block_pos[2] + env._baxter.torusRad) > block_pos[2]

        if y_bool and z_bool and x_bool:
            env.logger.debug(
                "Block within the hole. block_pos: {0} torus_pos: {1}".format(
                    str(block_pos), str(torus_pos)))
            env.logger.debug("env._envStepCounter: %s" %
                              str(env._envStepCounter))
            reward = 1
        else:
            reward = 0

        env.logger.debug("Reward: %s \n" % str(reward))
        return reward

    def torus_distance_reward(env):
        """Calculates the reward for the episode.
        """

        torus_pos = np.array(
            p.getBasePositionAndOrientation(env._baxter.torusUid)[0])
        env.logger.debug("Reward torus position: %s" % str(torus_pos))

        if env._useBlock:
            block_pos = np.array(
                p.getBasePositionAndOrientation(env._baxter.blockUid)[0])
        else:
            block_pos = np.array(
                p.getLinkState(env._baxter.baxterUid, 26)[0])  # 26 or avg between 28 and 30

        # Caculate distance between the center of the torus and the block
        distance = np.linalg.norm(np.array(torus_pos) - np.array(block_pos))
        reward = -distance
        env.logger.debug("Distance: %s" % str(distance))

        x_bool = (torus_pos[0] + env._baxter.margin) < block_pos[0]
        y_bool = (
            torus_pos[1] - env._baxter.torusRad) < block_pos[1] and (torus_pos[1] + env._baxter.torusRad) > block_pos[1]
        z_bool = (
            torus_pos[2] - env._baxter.torusRad) < block_pos[2] and (block_pos[2] + env._baxter.torusRad) > block_pos[2]

        cp_list = p.getContactPoints(
            env._baxter.baxterUid, env._baxter.torusUid)
        if any(cp_list):
            reward += env._collision_pen

        if y_bool and z_bool and x_bool:
            env.logger.debug(
                "Block within the hole. block_pos: {0} torus_pos: {1}".format(
                    str(block_pos), str(torus_pos)))
            env.logger.debug("env._envStepCounter: %s" %
                              str(env._envStepCounter))
            reward = reward + 10000

        env.logger.debug("Reward: %s \n" % str(reward))
        return reward

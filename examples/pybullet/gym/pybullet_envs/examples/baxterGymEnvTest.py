# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv
import time


def main():

    environment = BaxterGymEnv(
        renders=True, isDiscrete=False, maxSteps=10000000)

    motorsIds = []

    dv = 1
    min = 0
    max = 2
    motorsIds.append(environment._p.addUserDebugParameter("s0", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("s1", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("e0", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("e1", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("w0", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("w1", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("w2", -dv, dv, 0))

    done = False
    while (not done):

        action = []
        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))

        state, reward, done, info = environment.step(action)
        obs = environment.getExtendedObservation()


if __name__ == "__main__":
    main()

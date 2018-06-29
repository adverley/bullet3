# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import argparse
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_envs.bullet.baxter import Baxter
import numpy as np
import pybullet as p
from tqdm import tqdm

def main(args):

    if args.mode == 'force':
        p.connect(p.DIRECT)

        baxter = Baxter()
        max_x = -5
        max_y = -5
        max_z = -5
        max_pointX = None
        max_pointY = None
        max_pointZ = None

        print("Starting test run...")

        for s0 in tqdm(np.arange(-1.0, 1.0, 0.1)):
            for s1 in np.arange(-1.0, 1.0, 0.1):
                for e0 in np.arange(-1.0, 1.0, 0.1):
                    for e1 in np.arange(-1.0, 1.0, 0.1):
                        for w0 in np.arange(-1.0, 1.0, 0.1):
                            for w1 in np.arange(-1.0, 1.0, 0.1):
                                for w2 in np.arange(-1.0, 1.0, 0.1):
                                    #print("Test run:", w2)
                                    action = [s0, s1, e0, e1, w0, w1, w2]
                                    baxter.applyAction(action)
                                    temp = baxter.getEndEffectorPos()
                                    if temp[0] > max_x:
                                        max_pointX = temp
                                        max_x = temp[0]
                                    elif temp[1] > max_y:
                                        max_pointY = temp
                                        max_y = temp[1]
                                    elif temp[2] > max_z:
                                        max_pointZ = temp
                                        max_z = temp[2]

        print("Max X coordinate: ", max_pointX)
        print("Max Y coordinate: ", max_pointY)
        print("Max Z coordinate: ", max_pointZ)

    elif args.mode == 'manual':
        try:
            p.connect(p.GUI)
            baxter = Baxter()
            max_x = -5
            max_y = -5
            max_z = -5
            min_x = 5
            min_y = 5
            min_z = 5

            max_pointX = None
            max_pointY = None
            max_pointZ = None
            min_pointX = None
            min_pointY = None
            min_pointZ = None

            motorsIds = []
            dv = 1
            min = 0
            max = 2
            motorsIds.append(p.addUserDebugParameter("s0", -dv, dv, 0))
            motorsIds.append(p.addUserDebugParameter("s1", -dv, dv, 0))
            motorsIds.append(p.addUserDebugParameter("e0", -dv, dv, 0))
            motorsIds.append(p.addUserDebugParameter("e1", -dv, dv, 0))
            motorsIds.append(p.addUserDebugParameter("w0", -dv, dv, 0))
            motorsIds.append(p.addUserDebugParameter("w1", -dv, dv, 0))
            motorsIds.append(p.addUserDebugParameter("w2", -dv, dv, 0))

            done = False
            while (not done):
                action = []
                for motorId in motorsIds:
                    action.append(p.readUserDebugParameter(motorId))

                baxter.applyAction(action)
                obs = baxter.getEndEffectorPos()

                # max
                if obs[0] > max_x:
                    max_pointX = obs
                    max_x = obs[0]
                elif obs[1] > max_y:
                    max_pointY = obs
                    max_y = obs[1]
                elif obs[2] > max_z:
                    max_pointZ = obs
                    max_z = obs[2]

                # min
                if obs[0] < min_x:
                    min_pointX = obs
                    min_x = obs[0]
                elif obs[1] < min_y:
                    min_pointY = obs
                    min_y = obs[1]
                elif obs[2] < min_z:
                    min_pointZ = obs
                    min_z = obs[2]

                p.stepSimulation()

        except KeyboardInterrupt:
            print('Interrupted')
            print("Max X coordinate: ", max_pointX, "Min X coordinate: ", min_pointX)
            print("Max Y coordinate: ", max_pointY, "Min Y coordinate: ", min_pointY)
            print("Max Z coordinate: ", max_pointZ, "Min Z coordinate: ", min_pointZ)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['force', 'manual'], default='manual')
    args = parser.parse_args()
    main(args)

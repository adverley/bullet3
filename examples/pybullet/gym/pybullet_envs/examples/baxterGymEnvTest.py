# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import argparse
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv
import time


def main(args):

    environment = BaxterGymEnv(
                    renders=True,
                    maxSteps=10000000,
                    useCamera=False,
                    useBlock=True,
                    _reward_function=None,
                    _action_type=args.action_type
                )

    if args.action_type == 'continuous':
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

    elif args.action_type == 'single':
        motorsIds = []

        motorsIds.append(environment._p.addUserDebugParameter("motor", 0, 7, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dv", 0, 2, 1))

        done = False
        while (not done):

            action = []
            for motorId in motorsIds:
                action.append(round(environment._p.readUserDebugParameter(motorId)))

            action = action[0] + action[1]*7

            state, reward, done, info = environment.step(action)
            obs = environment.getExtendedObservation()

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-type', choices=['single', 'discrete', 'continuous'], default='continuous')
    args = parser.parse_args()
    main(args)

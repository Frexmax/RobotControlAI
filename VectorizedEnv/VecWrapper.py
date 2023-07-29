import pickle
import cloudpickle
import numpy as np

from Worker import worker
from multiprocessing import Pipe, Process
from GripperRiseEnv import GripperRiseEnv
from GripperPushEnv import GripperPushEnv
from GripperReachPositionEnv import GripperReachPositionEnv
from GripperRiseEvnMutliNetwork import GripperRiseEnvMultiNetwork


class CloudPickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


class VecWrapper:
    def __init__(self, env_fns, env_info):
        self.waiting = False
        self.closed = False
        self.observation_space = env_info["observation_space"]
        self.action_space = env_info["action_space"]
        self.goal_size = env_info["goal_size"]
        self.goal_limits_x = env_info["goal_limits"][0]
        self.goal_limits_y = env_info["goal_limits"][1]
        self.goal_limits_z = env_info["goal_limits"][2]

        no_of_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(no_of_envs)])
        self.ps = []

        for work, remote, function in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(target=worker, args=(work, remote, CloudPickleWrapper(function)))
            self.ps.append(process)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        if self.waiting:
            raise ValueError("Already Stepping")
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

    def step_wait(self):
        if not self.waiting:
            raise ValueError("Not Stepping")
        self.waiting = False
        results = [remote.recv() for remote in self.remotes]
        observations, rewards, dones = zip(*results)
        return observations, rewards, dones

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def compute_rewards(self, achieved_goal, substitute_goal, gripper_pos, cube_velocity=0):
        self.remotes[0].send(("compute_rewards", [achieved_goal, substitute_goal, gripper_pos, cube_velocity]))
        substitute_reward = self.remotes[0].recv()
        return substitute_reward

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(("close", None))

        for p in self.ps:
            p.join()
        self.closed = True


def make_mp_envs(env_id, num_env, reward_type="dense", dynamic_reward=False, version=1, stage=0, start_idx=0):
    def make_env(rank):
        def fn():
            if env_id == "rise":
                env1 = GripperRiseEnv(reward_type=reward_type, connection_type="direct",
                                      dynamic_reward=dynamic_reward, version=version)
            elif env_id == "push":
                env1 = GripperPushEnv(reward_type=reward_type, connection_type="direct")
            elif env_id == "reach":
                env1 = GripperReachPositionEnv(reward_type=reward_type, connection_type="direct")
            else:
                env1 = GripperRiseEnvMultiNetwork(reward_type=reward_type, connection_type="direct", train_stage=stage)
            return env1
        return fn

    if env_id == "rise":
        env_test = GripperRiseEnv(reward_type=reward_type, connection_type="direct",
                                  dynamic_reward=dynamic_reward, version=version)
    elif env_id == "push":
        env_test = GripperPushEnv(reward_type=reward_type, connection_type="direct")
    elif env_id == "reach":
        env_test = GripperReachPositionEnv(reward_type=reward_type, connection_type="direct")
    else:
        env_test = GripperRiseEnvMultiNetwork(reward_type=reward_type, connection_type="direct", train_stage=stage)

    data = {"observation_space": env_test.observation_space, "action_space": env_test.action_space,
            "goal_size": env_test.goal_size, "goal_limits":
                [env_test.goal_limits_x, env_test.goal_limits_y, env_test.goal_limits_z]}
    return VecWrapper([make_env(i + start_idx) for i in range(num_env)], data)

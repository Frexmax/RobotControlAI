import numpy as np
import pickle
import time

from PrioritizedBuffer import PrioritizedBuffer
from Buffer import Buffer
from datetime import date
from numba import jit

# https://github.com/hemilpanchiwala/Hindsight-Experience-Replay/blob/main/ddpg_with_her/DDPG_HER_main.py


class HerBufferV2:
    def __init__(self, env, env_name, state_shape, action_shape, num_workers, buffer_capacity=100_000,
                 batch_size=64, n_sampled_goals=8, internal_buffer_type="priority"):
        self.env = env
        self.goal_x_range = self.env.goal_limits_x
        self.goal_y_range = self.env.goal_limits_y
        self.goal_z_range = self.env.goal_limits_z

        self.env_name = env_name
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_workers = num_workers

        self.temp_episode_experience = [[] for i in range(self.num_workers)]
        self.n_sampled_goals = n_sampled_goals
        self.buffer_counter = 0
        self.internal_buffer_type = internal_buffer_type

        # 100 - MAX LEN OF EPISODE IN GRIPPER ENV
        self.observations = np.zeros((self.num_workers, 100, self.state_shape - 3), dtype=np.float32)
        self.n_observations = np.zeros((self.num_workers, 100, self.state_shape - 3), dtype=np.float32)
        self.actions = np.zeros((self.num_workers, 100, self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.num_workers, 100), dtype=np.float32)
        self.dones = np.zeros((self.num_workers, 100), dtype=np.bool_)

        self.achieved_goals = np.zeros((self.num_workers, 100, 3), dtype=np.float32)
        self.desired_goals = np.zeros((self.num_workers, 100, 3), dtype=np.float32)
        self.gripper_pos = np.zeros((self.num_workers, 100, 3), dtype=np.float32)
        self.worker_fills = np.zeros(self.num_workers, dtype=np.int32)

        if self.internal_buffer_type == "priority":
            self.sample_buffer = PrioritizedBuffer(self.env_name, state_shape, action_shape, buffer_capacity, batch_size)
        else:
            self.sample_buffer = Buffer(self.env_name, state_shape, action_shape, buffer_capacity, batch_size)

    @staticmethod
    @jit(nopython=True)
    def get_future_index(t_index, episode_length):
        return np.random.randint(t_index, episode_length)

    @staticmethod
    @jit(nopython=True)
    def concatenate_input(ob, ob_n, substitute_goal):
        inputs_her = np.concatenate((ob, substitute_goal))
        new_inputs_her = np.concatenate((ob_n, substitute_goal))
        return inputs_her, new_inputs_her

    def record(self, obs_tuple, worker_index):
        index = self.worker_fills[worker_index]
        self.observations[worker_index][index] = obs_tuple[0]
        self.n_observations[worker_index][index] = obs_tuple[1]
        self.actions[worker_index][index] = obs_tuple[2]
        self.rewards[worker_index][index] = obs_tuple[3]
        self.dones[worker_index][index] = obs_tuple[4]
        self.achieved_goals[worker_index][index] = obs_tuple[5]
        self.gripper_pos[worker_index][index] = obs_tuple[6]
        self.desired_goals[worker_index][index] = obs_tuple[7]
        self.worker_fills[worker_index] += 1

        inputs_real = np.concatenate([obs_tuple[0], obs_tuple[7]], axis=0)
        new_inputs_real = np.concatenate([obs_tuple[1], obs_tuple[7]], axis=0)
        self.sample_buffer.record((inputs_real, obs_tuple[2], obs_tuple[3], new_inputs_real, obs_tuple[4]))

        # self.temp_episode_experience[worker_index].append(obs_tuple)

    def check_goals(self, substitute_goals, original_goal, len_episode):
        for i in range(len_episode):
            goal = substitute_goals[i]
            if not (self.goal_x_range[0] <= goal[0] >= self.goal_x_range[1] and
                    self.goal_y_range[0] <= goal[1] >= self.goal_y_range[1] and
                    self.goal_z_range[0] <= goal[2] >= self.goal_z_range[1]):
                substitute_goals[i] = original_goal
        return substitute_goals

    def get_velocity_data(self, worker_index, len_episode):
        velocity_data = np.zeros((len_episode, 3), dtype=np.float32)
        for i in range(len_episode):
            velocity_data[i] = self.observations[worker_index][i][21:24]
        return velocity_data

    def update_buffer(self, worker_indices):
        for worker_index in worker_indices:
            if self.worker_fills[worker_index] != 0:
                len_episode = self.worker_fills[worker_index]
                original_goal = self.desired_goals[worker_index][-1]
                velocity_data = self.get_velocity_data(worker_index, len_episode)
                gripper_pos = self.gripper_pos[worker_index][0:len_episode]
                achieved_goals = self.achieved_goals[worker_index][0:len_episode]
                for k in range(self.n_sampled_goals):
                    indexes = np.arange(len_episode)
                    future = np.random.randint(indexes, len_episode, size=len_episode)
                    substitute_goals = self.achieved_goals[worker_index][future]
                    # substitute_goals = self.check_goals(substitute_goals, original_goal, len_episode)
                    substitute_rewards = self.env.compute_rewards(achieved_goals, substitute_goals,
                                                                  gripper_pos, velocity_data)

                    inputs_her = np.concatenate([self.observations[worker_index][0:len_episode],
                                                 substitute_goals], axis=1)
                    new_inputs_her = np.concatenate([self.n_observations[worker_index][0:len_episode],
                                                     substitute_goals], axis=1)

                    for t in range(len_episode):
                        self.sample_buffer.record((inputs_her[t], self.actions[worker_index][t], substitute_rewards[t],
                                                   new_inputs_her[t], self.dones[worker_index][t]))
                        self.buffer_counter += 1
                self.worker_fills[worker_index] = 0
                self.temp_episode_experience[worker_index] = []

    def get_mini_batch(self):
        return self.sample_buffer.get_mini_batch()

    def update_priorities(self, batch_indices, td):
        self.sample_buffer.update_priorities(batch_indices, td)

    def save(self):
        today = date.today()
        elements = np.min(self.buffer_capacity, self.buffer_counter)
        file_name = f"buffer-her-with-{self.internal_buffer_type}-month-{today.month}-day-{today.day}-elements-{elements}"
        path = f"Trainer/TrainDDPG/SavedModels/SavedBuffers/{file_name}"
        with open(f"{path}/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

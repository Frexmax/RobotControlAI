#  https://github.com/evilsocket/stable-baselines/blob/master/stable_baselines/common/running_mean_std.py

import numpy as np
from numba import jit


class StateNormalization:
    def __init__(self, observation_shape, normalize, batch_size, epsilon_count=1e-4, epsilon=1e-8, max_observation=10):
        self.mean = np.zeros(observation_shape, dtype=np.float64)
        self.var = np.zeros(observation_shape, dtype=np.float64)
        self.count = epsilon_count
        self.epsilon = epsilon
        self.max_clip = max_observation
        self.min_clip = -max_observation
        self.normalize = normalize
        self.batch_size = batch_size

        self.state_array = np.zeros((self.batch_size, observation_shape))
        self.index = 0

    def fit(self, state):

        if self.normalize:
            self.state_array[self.index] = state
            self.index += 1
            if self.index == self.batch_size:

                new_mean = np.mean(self.state_array, axis=0)
                new_var = np.var(self.state_array, axis=0)
                state_count = self.batch_size
                self.mean, self.var, self.count = self.update_from_states(new_mean, new_var, state_count,
                                                                          self.mean, self.var, self.count)
                self.state_array.fill(0)
                self.index = 0

    @staticmethod
    @jit(nopython=True)
    def update_from_states(batch_mean, batch_var, batch_count, current_mean, current_var, current_count):
        delta = batch_mean - current_mean
        total_count = batch_count + current_count
        new_mean = current_mean + delta * batch_count / total_count

        m_a = current_var * current_count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * current_count * batch_count / total_count
        new_var = m_2 / total_count
        return new_mean, new_var, total_count

    def get_normalized(self, state):
        if self.normalize:
            return np.clip((state - self.mean) / np.sqrt(self.var + self.epsilon), self.min_clip, self.max_clip)
        else:
            return state

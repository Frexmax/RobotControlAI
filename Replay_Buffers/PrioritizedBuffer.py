import numpy as np
import tensorflow as tf
import numba as nb
import pickle

from numba import jit, int64, float32
from datetime import date


class PrioritizedBuffer:
    def __init__(self, env_name, state_shape, action_shape, buffer_capacity=100000,
                 batch_size=64, alpha=0.7, beta=0.4, update_rate=50_000):

        self.env_name = env_name
        self.alpha = alpha
        self.beta = beta
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.alpha_update = 0.025
        self.beta_update = 0.025
        self.updated_parameters = False

        self.mask = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.probabilities = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.priorities = np.ones(self.buffer_capacity, dtype=np.float32)

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, state_shape), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_shape), dtype=np.float32)
        self.dones_buffer = np.full(self.buffer_capacity, False)

    def record(self, obs_tuple):

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.dones_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

        self.priorities[index] = max(self.priorities)
        if self.updated_parameters and (self.buffer_counter % self.update_rate == 0):
            self.update_hyperparameters()

    def update_hyperparameters(self):
        if self.alpha > 0.5:
            self.alpha -= self.alpha_update

        if self.beta < 0.6:
            self.beta += self.beta_update

    def update_priorities(self, batch_indices, td):
        self.priorities[batch_indices] = abs(td)[0]

    @staticmethod
    @jit(nb.types.Tuple((nb.types.Array(float32, 1, "C"), nb.types.Array(float32, 1, "C"),
                         nb.types.Array(float32, 1, "C"), int64))
         (int64, int64, float32, float32, nb.types.Array(float32, 1, "C"), nb.types.Array(float32, 1, "C")),
         nopython=True, fastmath=True)
    def calculate_w_p(buffer_counter, buffer_capacity, alpha, beta, priorities, probabilities):
        index_range = np.minimum(buffer_counter, buffer_capacity)
        buffer_priorities = priorities[:index_range] ** alpha
        buffer_sum = np.sum(buffer_priorities)
        probabilities[:index_range] = buffer_priorities / buffer_sum

        weights = (index_range * probabilities[:index_range]) ** -beta
        weights = weights / weights.max()

        return priorities, probabilities, weights, index_range

    def get_mini_batch(self):

        priorities, probabilities, weights, index_range = self.calculate_w_p(self.buffer_counter, self.buffer_capacity,
                                                                             self.alpha, self.beta, self.priorities,
                                                                             self.probabilities)

        batch_indices = np.random.choice(index_range, self.batch_size, p=self.probabilities[:index_range])

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.dones_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(tf.cast(done_batch, dtype=tf.float32))

        weights = tf.convert_to_tensor(weights[batch_indices], dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, batch_indices

    def save(self):
        today = date.today()
        elements = np.min(self.buffer_capacity, self.buffer_counter)
        file_name = f"buffer-prioritized-month-{today.month}-day-{today.day}-elements-{elements}"
        path = f"Trainer/TrainDDPG/SavedModels/SavedBuffers/{file_name}"
        with open(f"{path}/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


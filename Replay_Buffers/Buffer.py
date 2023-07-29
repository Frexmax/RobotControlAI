import numpy as np
import tensorflow as tf
import pickle
from datetime import date


class Buffer:
    def __init__(self, env_name, state_shape, action_shape, buffer_capacity=100000, batch_size=64):
        self.env_name = env_name
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, state_shape))
        self.action_buffer = np.zeros((self.buffer_capacity, action_shape))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_shape))
        self.dones_buffer = np.full(self.buffer_capacity, False)

    def record(self, obs_tuple):

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.dones_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    def get_mini_batch(self):

        index_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(index_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.dones_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(tf.cast(done_batch, dtype=tf.float32))

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, 1, batch_indices

    def save(self):
        today = date.today()
        elements = np.min(self.buffer_capacity, self.buffer_counter)
        file_name = f"buffer-uniform-month-{today.month}-day-{today.day}-elements-{elements}"
        path = f"Trainer/TrainDDPG/SavedModels/SavedBuffers/{file_name}"
        with open(f"{path}/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

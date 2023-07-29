import numpy as np
import tensorflow as tf


# https://towardsdatascience.com/hindsight-experience-replay-her-implementation-92eebab6f653


class HerBuffer:
    def __init__(self, env, state_shape, action_shape, buffer_capacity=100000, batch_size=64, n_sampled_goal=8,
                 selection_strategy="final"):
        self.env = env
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.her_ratio = 1 - (1.0 / (n_sampled_goal + 1))

        self.buffer_counter = 0
        self.episode = 0

        self.state_buffer = np.zeros((self.buffer_capacity, state_shape), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_capacity, dtype=bool)

        self.desired_goals = np.zeros((self.buffer_capacity, env.goal_size), dtype=np.float32)
        self.next_achieved_goals = np.zeros((self.buffer_capacity, env.goal_size), dtype=np.float32)

        self.episode_end_indices = np.zeros(self.buffer_capacity, dtype=np.int32)
        self.index_episode_map = np.zeros(self.buffer_capacity, dtype=np.int32)

        self.goal_selection_strategy = selection_strategy

    def record(self, obs_tuple):

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.desired_goals[index] = obs_tuple[3]  # TARGET POSITION
        self.next_achieved_goals[index] = obs_tuple[4]  # ACHIEVED POSITION
        self.done_buffer[index] = obs_tuple[5]
        self.index_episode_map[index] = obs_tuple[6]
        self.state_buffer[(index + 1) % self.buffer_capacity] = obs_tuple[7]

        self.buffer_counter += 1
        if obs_tuple[5]:
            self.episode_end_indices[self.episode] = index
            self.episode += 1

    def sample_trajectories(self,  batch_indices):
        her_batch_size = int(self.batch_size * self.her_ratio)
        her_indices = batch_indices[:her_batch_size]
        replay_indices = batch_indices[her_batch_size:]

        her_goals = self.sample_goals(her_indices)
        her_rewards = self.env.compute_rewards(self.next_achieved_goals[her_indices], her_goals)

        desired_goals = np.concatenate([her_goals, self.desired_goals[replay_indices]])
        rewards = np.concatenate([her_rewards, self.reward_buffer[replay_indices]])

        state = np.concatenate([self.state_buffer[batch_indices], desired_goals], axis=1)
        next_state = np.concatenate([self.state_buffer[(batch_indices + 1) % self.buffer_capacity],
                                     desired_goals], axis=1)

        actions = self.action_buffer[batch_indices]
        dones = self.done_buffer[batch_indices]

        state_batch = tf.convert_to_tensor(state)
        action_batch = tf.convert_to_tensor(actions)
        reward_batch = tf.convert_to_tensor(rewards)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state)

        done_batch = tf.convert_to_tensor(dones)
        done_batch = tf.convert_to_tensor(tf.cast(done_batch, dtype=tf.float32))

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, 1, np.concatenate([replay_indices, her_indices])

    def sample_goals(self, her_indices):
        her_episodes = self.index_episode_map[her_indices]
        episode_end_indices = self.episode_end_indices[her_episodes]

        if self.goal_selection_strategy == "final":
            goal_indices = episode_end_indices - 1

        elif self.goal_selection_strategy == "variable":
            goals = []
            reached_positions = self.next_achieved_goals[her_indices]
            max_pos = np.array(self.env.max_goal_area)
            min_pos = np.array(self.env.min_goal_area)

            for i, pos in enumerate(reached_positions):
                if (min_pos[0] <= pos[0] <= max_pos[0] and
                        min_pos[1] <= pos[1] <= max_pos[1] and
                        min_pos[2] <= pos[2] <= max_pos[2] and
                        min_pos[3] - 0.05 <= pos[1] <= max_pos[1] + 0.05):

                    goals.append(reached_positions[i])
                else:
                    goals.append(self.desired_goals[her_indices[i]])
            return np.array(goals)

        elif self.goal_selection_strategy == "future":
            found = False
            for i in range(len(her_indices)):
                if episode_end_indices[i] <= her_indices[i]:
                    found = True
                    break

            if found:
                goal_indices = episode_end_indices - 1
            else:
                goal_indices = np.random.randint(her_indices, episode_end_indices)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported.")

        return self.next_achieved_goals[goal_indices]

    def get_mini_batch(self):
        end_idx = self.episode_end_indices[self.episode - 1]

        if self.buffer_counter >= self.buffer_capacity:
            batch_indices = (np.random.randint(1, self.buffer_counter, size=self.batch_size) + end_idx) \
                         % self.buffer_capacity
        else:
            batch_indices = np.random.randint(0, end_idx, size=self.batch_size)

        return self.sample_trajectories(batch_indices)

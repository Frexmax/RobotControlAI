import numpy as np
import tensorflow as tf
import pickle

from HindSightBufferV2 import HerBufferV2
from PrioritizedBuffer import PrioritizedBuffer
from Buffer import Buffer
from OuNoise import OuNoise
from TrainerHelpers import create_env, create_model
from numba import jit

from TrainDDPG import train
from RewardEpsilonExploration import RewardEpsilonExploration
from StateNormalization import StateNormalization
from logger import Logger
from datetime import date
from VecWrapper import make_mp_envs
from pathlib import Path


class Trainer:
    def __init__(self, parameters):

        # ----------------------------- ENV PARAMETERS ----------------------------- #

        self.env_name = parameters["env_name"]
        self.version = parameters["version"]
        self.num_workers = parameters["num_workers"]
        self.stage = parameters["stage"]
        self.env_reward_stage = parameters["env_reward_stage"]
        self.reward_type = parameters["reward_type"]
        self.connection_type = parameters["connection_type"]
        self.dynamic_reward = parameters["dynamic_reward"]

        self.env = self.generate_env()  # CREATE ENV SINGLE OR MULTI WORKER
        self.total_observation_space = self.env.observation_space + self.env.goal_size

        # ----------------------------- TRAINER PARAMETERS ----------------------------- #

        self.show_every = parameters["show_every"]
        self.save_every = parameters["save_every"]
        self.time_steps = parameters["num_time_steps"]
        self.test_mode = parameters["test_mode"]
        self.save_every_buffer = parameters["save_every_buffer"]
        self.save_threshold = float("-inf")
        self.load = parameters["load"]

        # ----------------------------- EXPLORATION PARAMETERS ----------------------------- #

        self.start_epsilon = parameters["start_epsilon"]
        self.end_epsilon = parameters["end_epsilon"]
        self.min_reward = parameters["min_reward"]
        self.max_reward = parameters["max_reward"]
        self.update_steps = parameters["update_steps"]
        self.update_every = parameters["update_every"]

        self.epsilon_exploration = RewardEpsilonExploration(self.start_epsilon, self.end_epsilon, self.min_reward,
                                                            self.max_reward, self.update_steps)  # EPSILON EXPLORATION

        # ----------------------------- MODEL PARAMETERS ----------------------------- #

        self.model_type = parameters["model_type"]
        self.batch_size = parameters["batch_size"]
        self.critic_lrn = parameters["critic_lrn"]
        self.actor_lrn = parameters["actor_lrn"]
        self.discount_factor = parameters["discount_factor"]
        self.polyak = parameters["polyak"]

        if self.env_name == "rise_multi" and self.stage == 1:
            self.critic_path_1, self.critic_path_2 = parameters["model_paths-1"][0], parameters["model_paths-2"][0]
            self.actor_path_1, self.actor_path_2 = parameters["model_paths-1"][1], parameters["model_paths-2"][1]
        else:
            self.critic_path = parameters["model_paths"][0]
            self.actor_path = parameters["model_paths"][1]

        models = self.generate_model()
        if self.env_name == "rise_multi" and self.stage == 1:
            self.online_value_model_1, self.target_value_model_1, self.online_policy_model_1, self.target_policy_model_1, \
            self.online_value_model, self.target_value_model, self.online_policy_model, self.target_policy_model = models
        else:
            self.online_value_model, self.target_value_model, self.online_policy_model, self.target_policy_model = models
        self.value_optimizer = tf.keras.optimizers.Adam(self.critic_lrn)
        self.policy_optimizer = tf.keras.optimizers.Adam(self.actor_lrn)

        # ----------------------------- BUFFER PARAMETERS ----------------------------- #

        self.buffer_type = parameters["buffer_type"]
        self.her_strategy = parameters["her_strategy"]
        self.memory_size = parameters["memory_size"]
        self.n_sample_goals = parameters["n_sample_goals"]
        self.internal_buffer_type = parameters["internal_buffer_type"]

        self.buffer = self.generate_buffer()  # CREATE BUFFER

        # ----------------------------- NOISE PARAMETERS ----------------------------- #

        self.ou_std = parameters["ou_std"]

        self.ou_noise = OuNoise(mean=np.zeros(4), std_deviation=self.ou_std * np.ones(4))  # CREATE NOISE

        # ----------------------------- NORMALIZER PARAMETERS ----------------------------- #

        self.normalize = parameters["normalize"]

        self.state_normalizer = StateNormalization(self.env.observation_space, self.normalize, self.batch_size)
        self.goal_normalizer = StateNormalization(self.env.goal_size, self.normalize, self.batch_size)

        # ----------------------------- LOGGER PARAMETERS ----------------------------- #

        self.logger_path = parameters["logger_path"]

        self.logger = self.generate_logger()  # CREATE LOGGER

    def generate_env(self):
        if self.num_workers > 1:
            env = make_mp_envs(self.env_name, self.num_workers, self.reward_type,
                               self.dynamic_reward, self.version, self.stage)
        else:
            env = create_env(self.env_name, self.reward_type, self.connection_type,
                             self.dynamic_reward, self.version, self.stage)
        return env

    def generate_buffer(self):
        if self.buffer_type == "her":
            buffer = HerBufferV2(self.env, self.env_name, self.total_observation_space,
                                 self.env.action_space, self.num_workers, self.memory_size,
                                 self.batch_size, self.n_sample_goals, self.internal_buffer_type)
        elif self.buffer_type == "priority":
            buffer = PrioritizedBuffer(self.env_name, self.total_observation_space,
                                       self.env.action_space, self.memory_size, self.batch_size)
        else:
            buffer = Buffer(self.env_name, self.total_observation_space,
                            self.env.action_space, self.memory_size, self.batch_size)
        return buffer

    def generate_logger(self):
        if not self.load:
            logger = Logger(self.env_name, self.model_type)
        else:
            with open(self.logger_path, "rb") as logger_file:
                logger = pickle.load(logger_file)
        return logger

    def generate_model(self):
        if self.env_name == "rise_multi" and self.stage == 1:
            # 1st STAGE NETWORKS - NOT TRAINED
            online_value_model_1 = tf.keras.models.load_model(self.critic_path_1)
            target_value_model_1 = tf.keras.models.load_model(self.critic_path_1)
            online_policy_model_1 = tf.keras.models.load_model(self.actor_path_1)
            target_policy_model_1 = tf.keras.models.load_model(self.actor_path_1)

            # 2nd STAGE NETWORKS
            online_value_model = tf.keras.models.load_model(self.critic_path_2)
            target_value_model = tf.keras.models.load_model(self.critic_path_2)
            online_policy_model = tf.keras.models.load_model(self.actor_path_2)
            target_policy_model = tf.keras.models.load_model(self.actor_path_2)
            return online_value_model_1, target_value_model_1, online_policy_model_1, target_policy_model_1, \
                   online_value_model, target_value_model, online_policy_model, target_policy_model
        else:
            if not self.load:
                online_policy_model, online_value_model = create_model(self.total_observation_space,
                                                                       self.env.action_space,
                                                                       self.env_name, self.version)
                target_policy_model, target_value_model = create_model(self.total_observation_space,
                                                                       self.env.action_space,
                                                                       self.env_name, self.version)
                target_value_model.set_weights(online_value_model.get_weights())
                target_policy_model.set_weights(online_policy_model.get_weights())
            else:
                online_value_model = tf.keras.models.load_model(self.critic_path)
                target_value_model = tf.keras.models.load_model(self.critic_path)
                online_policy_model = tf.keras.models.load_model(self.actor_path)
                target_policy_model = tf.keras.models.load_model(self.actor_path)
            return online_value_model, target_value_model, online_policy_model, target_policy_model

    def create_neural_input(self, current_state):
        normalized_state = self.state_normalizer.get_normalized(current_state["observation"])
        normalized_goal = self.goal_normalizer.get_normalized(current_state["desired_goal"])
        neural_input = np.concatenate([normalized_state, normalized_goal], axis=1)
        return neural_input

    @staticmethod
    @tf.function
    def policy(policy_model, observations):
        return tf.squeeze(policy_model(observations))

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def random_action(num_workers, version, env_name):
        action_pos = (np.random.rand(num_workers, 3) * 2) - 1
        action_finger = np.random.rand(num_workers, 1) * 0.04
        if env_name == "rise" and version == 0:
            action_orientation = (np.random.rand(num_workers, 3) * 2) - 1
            action = np.concatenate((action_pos, action_finger, action_orientation), axis=1)
        else:
            action = np.concatenate((action_pos, action_finger), axis=1)
        return action

    def add_noise(self, action):
        noise = self.ou_noise()
        noise[3] *= 0.04
        action = action.numpy() + noise
        return action

    def graph_training(self):
        self.logger.graph_log()

    def reshape_state(self, state):
        state_n = {"observation": np.zeros((self.num_workers, self.env.observation_space), dtype=np.float32),
                   "achieved_goal": np.zeros((self.num_workers, self.env.goal_size), dtype=np.float32),
                   "desired_goal": np.zeros((self.num_workers, self.env.goal_size), dtype=np.float32)}

        if self.num_workers > 1:
            for worker in range(self.num_workers):
                state_n["observation"][worker] = state[worker]["observation"]
                state_n["achieved_goal"][worker] = state[worker]["achieved_goal"]
                state_n["desired_goal"][worker] = state[worker]["desired_goal"]
        else:
            state_n["observation"][0] = state["observation"]
            state_n["achieved_goal"][0] = state["achieved_goal"]
            state_n["desired_goal"][0] = state["desired_goal"]
        return state_n

    @staticmethod
    @jit(nopython=True, cache=True)
    def check_done(done):
        return np.where(done)

    @staticmethod
    @jit(nopython=True, cache=True)
    def update_done(complete_episodes, num_completed, episode_length, episode_rewards, num_episodes):

        complete_indexes = complete_episodes[0]
        num_reached = (episode_length[complete_indexes] < 100).astype(np.float64)
        num_rewards = episode_rewards[complete_indexes]
        episode_length[complete_indexes] = 0
        episode_rewards[complete_indexes] = 0
        num_episodes += num_completed
        return num_reached, num_rewards, episode_length, episode_rewards, num_episodes

    def reshape_data(self, reward, done, action):
        if self.num_workers == 1:
            return np.expand_dims(reward, axis=0), np.expand_dims(done, axis=0), np.expand_dims(action, axis=0)
        else:
            return reward, done, action

    def save_buffer(self):

        today = date.today()
        num_elements = np.clip(0, self.memory_size)
        buffer_path = f"buffer-{self.env_name}-month-{today.month}-day-{today.day}-" \
                      f"elements-{num_elements}-reward_type-{self.reward_type}"

        with open(f'saved_models/saved_buffers/{buffer_path}.pkl', 'wb') as buffer_file:
            pickle.dump(self.buffer, buffer_file, pickle.HIGHEST_PROTOCOL)

    def load_buffer(self, buffer_path):

        if Path(buffer_path).is_file():
            with open(buffer_path, 'wb') as buffer_file:
                self.buffer = pickle.load(buffer_file)
        else:
            self.fill_buffer()

    def train_normalizer(self):

        for normalizer_step in range(50_000 // self.num_workers):
            normalizer_action = np.squeeze(self.random_action(self.num_workers, self.version, self.env_name))
            new_state, reward, done = self.env.step(normalizer_action)
            new_state = self.reshape_state(new_state)

            for worker in range(self.num_workers):

                self.state_normalizer.fit(new_state['observation'][worker])
                self.goal_normalizer.fit(new_state['achieved_goal'][worker])
                self.goal_normalizer.fit(new_state['desired_goal'][worker])

            complete_episodes = self.check_done(np.array(done))
            num_completed = len(complete_episodes[0])
            if num_completed > 0:
                if self.num_workers == 1:
                    _ = self.env.reset()

    def fill_buffer(self):

        current_state = self.reshape_state(self.env.reset())
        current_env_stage = 0
        for fill_step in range(self.memory_size // (self.num_workers * 2)):
            if self.env_name == "rise_multi" and self.stage == 1:
                current_env_stage = self.env.current_stage

            if np.random.rand() > self.epsilon_exploration.epsilon:
                neural_input = self.create_neural_input(current_state)
                fill_action = self.policy(self.online_policy_model, neural_input)
            else:
                fill_action = np.squeeze(self.random_action(self.num_workers, self.version, self.env_name))

            new_state, reward, done = self.env.step(fill_action)
            reward, done, action = self.reshape_data(reward, done, fill_action)
            new_state = self.reshape_state(new_state)

            if self.env_name == "rise_multi" and self.stage == 1:
                if current_env_stage == 1:
                    for worker in range(self.num_workers):
                        self.update_buffer(current_state, new_state, action, reward, done, worker)
            else:
                for worker in range(self.num_workers):
                    self.update_buffer(current_state, new_state, action, reward, done, worker)

            current_state = new_state
            complete_episodes = self.check_done(np.array(done))
            num_completed = len(complete_episodes[0])

            if num_completed > 0 and self.num_workers == 1:
                current_state = self.reshape_state(self.env.reset())

            if num_completed > 0:
                if self.buffer_type == "her":
                    self.buffer.update_buffer(complete_episodes[0])
                if self.num_workers == 1:
                    current_state = self.reshape_state(self.env.reset())

    def save_model_log(self, reach_rate):
        today = date.today()
        last_step = self.logger.info["steps"][-1]

        if self.dynamic_reward:
            stage = self.env.reward_stage
        else:
            stage = 3

        policy_name = f"{self.env_name}-{self.model_type}-policy-stage-{stage}-month-{today.month}-day-" \
                      f"{today.day}-{last_step}-{int(reach_rate)}%-{self.reward_type}"

        value_name = f"{self.env_name}-{self.model_type}-value-stage-{stage}-month-{today.month}-day-" \
                     f"{today.day}-{last_step}-{int(reach_rate)}%-{self.reward_type}"

        if self.env_name == "reach":
            self.online_policy_model.save(f'Trainer/TrainDDPG/SavedModels/ReachModels/{policy_name}.h5')
            self.online_value_model.save(f'Trainer/TrainDDPG/SavedModels/ReachModels/{value_name}.h5')
        elif self.env_name == "push":
            self.online_policy_model.save(f'Trainer/TrainDDPG/SavedModels/PushModels/{policy_name}.h5')
            self.online_value_model.save(f'Trainer/TrainDDPG/SavedModels/PushModels/{value_name}.h5')
        else:
            self.online_policy_model.save(f'Trainer/TrainDDPG/SavedModels/RiseModels/{policy_name}.h5')
            self.online_value_model.save(f'Trainer/TrainDDPG/SavedModels/RiseModels/{value_name}.h5')

        self.logger.save_log()

    def update_buffer(self, current_state, new_state, action, reward, done, worker=1):
        normalized_state = self.state_normalizer.get_normalized(current_state['observation'][worker])
        normalized_new_state = self.state_normalizer.get_normalized(new_state['observation'][worker])
        normalized_goal = self.goal_normalizer.get_normalized(current_state['desired_goal'][worker])

        if self.buffer_type == "her":
            normalized_achieved_goal = self.goal_normalizer.get_normalized(
                current_state['achieved_goal'][worker])
            self.buffer.record((normalized_state, normalized_new_state, action[worker], reward[worker],
                                done[worker], normalized_achieved_goal, normalized_state[0:3],
                                normalized_goal), worker)
        else:
            normalized_new_goal = self.goal_normalizer.get_normalized(new_state['desired_goal'][worker])
            self.buffer.record((np.concatenate([normalized_state, normalized_goal]), action[worker],
                                reward[worker], np.concatenate([normalized_new_state, normalized_new_goal]),
                                done[worker]))

    def run(self):

        # ----------------------------- TRAINING ----------------------------- #

        ep_rewards = []
        reaches = []
        training_loss = []

        episode_rewards = np.zeros(self.num_workers)
        episode_length = np.zeros(self.num_workers)
        current_state = self.reshape_state(self.env.reset())

        num_episodes = 0
        total_episodes = 0
        current_env_stage = 0

        for time_step in range(self.time_steps):

            if self.env_name == "rise_multi" and self.stage == 1:
                current_env_stage = self.env.current_stage

            # ----------------------------- PERFORM ACTION ----------------------------- #

            if self.env_name == "rise_multi" and self.stage == 1 and current_env_stage == 0:
                if np.random.rand() > 0.05:
                    neural_input = self.create_neural_input(current_state)
                    action = self.policy(self.online_policy_model_1, neural_input)
                    action = self.add_noise(action)
                else:
                    action = np.squeeze(self.random_action(self.num_workers, self.version, self.env_name))

            else:
                if np.random.rand() > self.epsilon_exploration.epsilon:
                    neural_input = self.create_neural_input(current_state)
                    action = self.policy(self.online_policy_model, neural_input)
                    action = self.add_noise(action)
                else:
                    action = np.squeeze(self.random_action(self.num_workers, self.version, self.env_name))

            new_state, reward, done = self.env.step(action)
            reward, done, action = self.reshape_data(reward, done, action)
            new_state = self.reshape_state(new_state)

            # ----------------------------- RECORD IN BUFFER ----------------------------- #

            if not self.test_mode:
                if self.env_name == "rise_multi" and self.stage == 1:
                    if current_env_stage == 1:
                        for worker in range(self.num_workers):
                            self.update_buffer(current_state, new_state, action, reward, done, worker)
                else:
                    for worker in range(self.num_workers):
                        self.update_buffer(current_state, new_state, action, reward, done, worker)\
            
                # ----------------------------- MODEL UPDATE ----------------------------- #
                
                if self.env_name == "rise_multi" and self.stage == 1:
                    if self.env.current_stage == 1:
                        loss_value, loss_policy, batch_indices, td = train(self.env, self.online_value_model,
                                                                           self.target_value_model,
                                                                           self.online_policy_model,
                                                                           self.target_policy_model, self.buffer,
                                                                           self.discount_factor, self.polyak,
                                                                           self.value_optimizer, self.policy_optimizer)
                        if len(batch_indices) == self.batch_size:
                            training_loss.append([loss_value, loss_policy])
                            if self.buffer_type == "priority" or self.internal_buffer_type == "priority":
                                self.buffer.update_priorities(batch_indices, td)
                else:
                    loss_value, loss_policy, batch_indices, td = train(self.env, self.online_value_model,
                                                                       self.target_value_model,
                                                                       self.online_policy_model,
                                                                       self.target_policy_model, self.buffer,
                                                                       self.discount_factor, self.polyak,
                                                                       self.value_optimizer, self.policy_optimizer)
                
                    if len(batch_indices) == self.batch_size:
                        training_loss.append([loss_value, loss_policy])
                        if self.buffer_type == "priority" or self.internal_buffer_type == "priority":
                            self.buffer.update_priorities(batch_indices, td)

            current_state = new_state
            episode_length += 1
            episode_rewards += reward

            complete_episodes = self.check_done(np.array(done))
            num_completed = len(complete_episodes[0])

            if num_completed > 0 and self.num_workers == 1:
                current_state = self.reshape_state(self.env.reset())

            if num_completed > 0 and not self.test_mode:
                if self.buffer_type == "her":
                    self.buffer.update_buffer(complete_episodes[0])
                done_update = self.update_done(complete_episodes, num_completed,
                                               episode_length, episode_rewards, num_episodes)

                reaches.extend(done_update[0])
                ep_rewards.extend(done_update[1])
                episode_length = done_update[2]
                episode_rewards = done_update[3]
                num_episodes = done_update[4]

                if num_episodes % self.update_every == 0 and time_step != 0:
                    self.epsilon_exploration.update_epsilon(
                        sum(ep_rewards[-self.update_every:]) / self.update_every)

                if self.num_workers == 1:
                    current_state = self.reshape_state(self.env.reset())

            # ----------------------------- PRINT LOG / SAVE MODELS ----------------------------- #

            if time_step % self.show_every == 0 and time_step != 0 and not self.test_mode:
                total_episodes += num_episodes
                average_reward = sum(ep_rewards) / num_episodes
                reach_rate = (sum(reaches) / num_episodes) * 100

                actor_loss = 0
                critic_loss = 0
                for i in range(len(training_loss)):
                    critic_loss += training_loss[i][0]
                    actor_loss += training_loss[i][1]
                average_critic_loss = float(critic_loss) / max(len(training_loss), 1)
                average_actor_loss = float(actor_loss) / max(len(training_loss), 1)

                if self.load:
                    self.logger.update_log(total_episodes, self.show_every, self.epsilon_exploration.epsilon,
                                           average_reward, [average_critic_loss, average_actor_loss], reach_rate)
                else:
                    self.logger.update_log(total_episodes, time_step, self.epsilon_exploration.epsilon,
                                           average_reward, [average_critic_loss, average_actor_loss], reach_rate)
                training_loss = []
                reaches = []
                ep_rewards = []
                num_episodes = 0

                self.logger.print_log(1)
                if time_step % self.save_every_buffer:
                    pass

                if average_reward > self.save_threshold and not self.dynamic_reward:
                    self.save_model_log(reach_rate)
                    self.save_threshold = average_reward

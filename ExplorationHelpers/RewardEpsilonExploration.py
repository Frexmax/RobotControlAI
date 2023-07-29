class RewardEpsilonExploration:
    def __init__(self, start_epsilon, end_epsilon, min_reward, max_reward, update_steps):
        self.epsilon = start_epsilon
        self.reward_threshold = min_reward
        self.reward_step = abs(min_reward - max_reward) / update_steps
        self.epsilon_step = abs(start_epsilon - end_epsilon) / update_steps

    def update_epsilon(self, episode_reward):
        if episode_reward > self.reward_threshold:
            self.epsilon -= self.epsilon_step
            self.reward_threshold += self.reward_step

    def reset_exploration(self, start_epsilon, end_epsilon, min_reward, max_reward, update_steps):
        self.epsilon = start_epsilon
        self.reward_threshold = min_reward
        self.reward_step = abs(min_reward - max_reward) / update_steps
        self.epsilon_step = abs(start_epsilon - end_epsilon) / update_steps

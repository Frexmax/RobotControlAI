import matplotlib.pyplot as plt
import pickle
from datetime import date


class Logger:
    def __init__(self, env_name, model_type="ddpg", goal_env=True):

        self.model_type = model_type
        self.goal_env = goal_env
        self.env_name = env_name
        self.num_items = 0

        self.info = {"episode": [], "steps": [], "epsilon": [], "reward": [], "loss": [], "reach_rate": []}
        # LOSS: Pos 0 - loss value, Pos 1 -  loss policy (DDPG)

    def update_log(self, episode, steps, epsilon, reward, loss, reach_rate):

        if self.num_items != 0 and self.info["episode"][-1] > episode:
            self.info["episode"].append(episode + self.info["episode"][-1])
        else:
            self.info["episode"].append(episode)

        if self.num_items != 0 and self.info["steps"][-1] > steps:
            self.info["steps"].append(steps + self.info["steps"][-1])
        else:
            self.info["steps"].append(steps)
        self.info["epsilon"].append(epsilon)
        self.info["reward"].append(reward)
        self.info["loss"].append(loss)
        self.info["reach_rate"].append(reach_rate)
        self.num_items += 1

    def print_log(self, last_elements):
        for key in self.info:
            if key == "steps":
                print(key.upper(), ":", self.info[key][-1])
            elif key == "reach_rate":
                print(key.upper(), ":", sum(self.info[key][-last_elements:]) / last_elements, "%")
            elif key == "loss" and self.model_type == "ddpg":
                critic_loss = 0
                actor_loss = 0
                for loss in self.info[key][-last_elements:]:
                    critic_loss += loss[0]
                    actor_loss += loss[1]

                print("CRITIC LOSS", ":", critic_loss / last_elements)
                print("ACTOR LOSS", ":", actor_loss / last_elements)
            else:
                print(key.upper(), ":", sum(self.info[key][-last_elements:]) / last_elements)
        print("")

    def graph_log(self):
        figure, axis = plt.subplots(2, 2)
        epsilon = axis[0, 0].plot(self.info["steps"], self.info["epsilon"], label="epsilon",
                                  color="orange", linewidth=2.0)
        axis[0, 0].set_xlabel('Steps')
        axis[0, 0].set_ylabel('Epsilon Value')
        axis[0, 0].set_title("Epsilon")
        axis[0, 0].set_ylim([-0.1, 1])
        axis[0, 0].legend()

        reward = axis[0, 1].plot(self.info["steps"], self.info["reward"], label='episode reward',
                                 color="lawngreen", linewidth=2.0)
        axis[0, 1].set_xlabel('Steps')
        axis[0, 1].set_ylabel('Reward per Episode')
        axis[0, 1].set_title("Reward")
        axis[0, 1].legend()

        ax2 = axis[1, 0].twinx()
        critic_loss_data = []
        actor_loss_data = []
        for loss in self.info["loss"]:
            critic_loss_data.append(loss[0])
            actor_loss_data.append(loss[1])

        critic_loss, = axis[1, 0].plot(self.info["steps"], critic_loss_data,
                                       label="critic_loss", color="red", linewidth=2.0)
        actor_loss, = ax2.plot(self.info["steps"], actor_loss_data,
                               label="actor_loss", color="blue", linewidth=2.0)

        axis[1, 0].set_xlabel('Steps')
        axis[1, 0].set_ylabel('Critic Loss Values')
        ax2.set_ylabel('Actor Loss Values')

        axis[1, 0].set_title("Actor Loss and Critic Loss")
        axis[1, 0].legend(handles=[critic_loss, actor_loss])

        reach_rate = axis[1, 1].plot(self.info["steps"], self.info["reach_rate"],
                                     color="green", label="reach rate", linewidth=2.0)
        axis[1, 1].set_xlabel('Steps')
        axis[1, 1].set_ylabel('Reach Rate (%)')
        axis[1, 1].set_ylim([-2, 100])
        axis[1, 1].set_title("Reach Rate")
        axis[1, 1].legend()

        plt.show()

    def save_log(self):
        today = date.today()
        file_name = f"log-{self.env_name}-{self.model_type}-month-{today.month}-day-{today.day}-ep-" \
                    f"{self.info['steps'][-1]}-{self.info['reach_rate'][-1]}%"
        with open(f"Logger/Logs/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def clear_log(self):
        self.info = {"episode": [], "steps": [], "epsilon": [], "reward": [], "loss": [], "reach_rate": []}
        self.num_items = 0

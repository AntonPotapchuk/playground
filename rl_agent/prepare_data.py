import numpy as np
import os

from keras.utils import to_categorical
from pommerman.agents import SimpleAgent
from pommerman.configs import ffa_competition_env
from pommerman.envs.v0 import Pomme

initial_rollouts = 1
train_data_path = './dagger/train_data/'
train_data_obs = 'obs.npy'
train_data_labels = 'labels.npy'
train_data_reward = 'reward.npy'
train_data_done = 'done.npy'
if not os.path.isdir(train_data_path):
    os.makedirs(train_data_path)


# Environment wrapper
class Stimulator:
    def __init__(self, env, config):
        self.env = env
        self.init(config)
        self.episode_number = 0

    def init(self, config):
        self.env.seed(0)
        # Add 3 random agents
        agents = []
        for agent_id in range(4):
            agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
        self.env.set_agents(agents)
        self.env.set_init_game_state(None)

    def stimulate(self, num_rollouts, render=False, logging=False):
        observations = []
        actions = []
        dones = []
        rewards = []

        for i in range(num_rollouts):
            self.episode_number += 1
            obs = self.env.reset()
            done = False
            episode_steps = 0

            while not done:
                if render:
                    self.env.render()
                all_actions = self.env.act(obs)
                obs, reward, done, _ = self.env.step(all_actions)
                episode_steps += 1

                observations.append(obs)
                actions.append(all_actions)
                rewards.append(reward)
                dones.append(done)

            print('rollout %i/%i' % (i + 1, num_rollouts))
        return np.array(observations), np.array(to_categorical(actions, self.env.action_space.n)), np.array(rewards), np.array(dones)

# Instantiate the environment
config = ffa_competition_env()
env = Pomme(**config["env_kwargs"])



# Generate training data
stimulator = Stimulator(env, config)
observations, actions, rewards, dones = stimulator.stimulate(num_rollouts=initial_rollouts)

np.save(train_data_path + train_data_obs, observations)
np.save(train_data_path + train_data_labels, actions)
np.save(train_data_path + train_data_reward, rewards)
np.save(train_data_path + train_data_done, dones)

#print(np.sum(training_data_labels, axis=0) / np.sum(training_data_labels))
import os
import numpy as np
import tensorflow as tf
from tensorforce.agents import PPOAgent

from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.characters import Bomber
from pommerman.configs import ffa_v0_env
from pommerman.constants import BOARD_SIZE
from pommerman.envs.v0 import Pomme
from rl_agent.pomm_network import PommNetwork

main_dir = './ppo/'
log_path = main_dir + 'logs/'
model_path = main_dir + 'model'

class TrainedAgent(BaseAgent):
    def __init__(self, actions, seed=0, save_path="./model/model.cptk", character=Bomber, model_path = model_path, batching_capacity = 1000):
        super(TrainedAgent, self).__init__(character=character)

        # Create a Proximal Policy Optimization agent
        network = dict(type='rl_agent.pomm_network.PommNetwork')
        states = {
            "board": dict(shape=(BOARD_SIZE, BOARD_SIZE, 3,), type='float'),
            "state": dict(shape=(3,), type='float')
        }
        saver = {
            "directory": model_path,
            "load": os.path.isdir(model_path)
        }
        self.agent = PPOAgent(
            states=states,
            actions=dict(type='int', num_actions=env.action_space.n),
            network=network,
            batching_capacity=batching_capacity,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-4
            ),
            saver=saver
        )

    @staticmethod
    def featurize(obs):
        def get_matrix(dict, key):
            res = dict[key]
            return res.reshape(res.shape[0], res.shape[1], 1).astype(np.float32)

        board = get_matrix(obs, 'board')
        teammate_position = None
        teammate = obs["teammate"]
        if teammate is not None:
            teammate = teammate.value
            if teammate > 10 and teammate < 15:
                teammate_position = np.argwhere(board == teammate)[0]
        else:
            teammate = None
        # My self - 11
        # Team mate - 12
        # Enemy - 13

        # Everyone enemy
        board[(board > 10) & (board < 15)] = 13
        # I'm not enemy
        my_position = obs['position']
        board[my_position[0], my_position[1], 0] = 11
        # Set teammate
        if teammate_position is not None:
            board[teammate_position[0], teammate_position[1], teammate_position[2]] = 12

        bomb_blast_strength = get_matrix(obs, 'bomb_blast_strength')
        bomb_life = get_matrix(obs, 'bomb_life')
        conv_inp = np.concatenate([board, bomb_blast_strength, bomb_life], axis=2)
        state = np.array([obs["ammo"], obs["blast_strength"], obs["can_kick"]]).astype(np.float32)
        return dict(board=conv_inp, state=state)

    def act(self, obs, action_space=None):
        obs = self.featurize(obs)
        res = self.agent.act(obs)
        if res==0:
            print("stop")
        elif res == 1:
            print("up")
        elif res == 2:
            print("down")
        elif res == 3:
            print("left")
        elif res == 4:
            print("right")
        elif res == 5:
            print("bomb")
        else:
            print("BAD")
        return res

    def close(self):
        self.sess.close()


# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)

# Add 3 random agents
agents = []
for agent_id in range(3):
    agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent
agent_id += 1
agents.append(TrainedAgent(env.action_space.n, character=config["agent"](agent_id, config["game_type"])))
env.set_agents(agents)
env.set_init_game_state(None)

# Seed and reset the environment
env.seed(0)
obs = env.reset()

# Run the agents until we're done
done = False
while not done:
    env.render()
    actions = env.act(obs)
    obs, reward, done, info = env.step(actions)
env.render(close=True)
env.close()

# Print the result
print(info)
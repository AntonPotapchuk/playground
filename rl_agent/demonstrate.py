import os
import numpy as np
import tensorflow as tf

from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.characters import Bomber
from pommerman.configs import ffa_v0_env
from pommerman.constants import BOARD_SIZE
from pommerman.envs.v0 import Pomme
from rl_agent.pomm_network import PommNetwork


class TrainedAgent(BaseAgent):
    def __init__(self, actions, seed=0, save_path="./model/model.cptk", character=Bomber):
        super(TrainedAgent, self).__init__(character=character)
        self.save_path = save_path
        self.actions = actions
        self.sess = tf.InteractiveSession()

        # TODO hardcoded
        self.conv_ph = tf.placeholder(shape=[None, BOARD_SIZE, BOARD_SIZE, 3], name='conv_ph', dtype=tf.float32)
        self.state_ph = tf.placeholder(shape=[None, 3], name='state_ph', dtype=tf.float32)

        network = PommNetwork.create_network({'board': self.conv_ph, 'state': self.state_ph})
        logits = tf.layers.dense(network, actions)
        self.sampled_action = tf.squeeze(tf.multinomial(logits, 1), axis=[1])
        self.saver = tf.train.Saver()
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.saver.restore(self.sess, self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")
                raise Exception("Model load failed")

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
        board = np.expand_dims(obs['board'], axis=0)
        state = np.expand_dims(obs['state'], axis=0)
        res = self.sess.run(self.sampled_action, feed_dict={self.conv_ph: board, self.state_ph: state})
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
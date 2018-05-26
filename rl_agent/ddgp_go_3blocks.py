
# coding: utf-8

# In[1]:


import numpy as np
import os
import tensorflow as tf
import warnings

from keras.layers import Dense, Input, Concatenate, Convolution2D, BatchNormalization, Activation, Flatten, Add
from keras.models import Model, load_model
from keras.optimizers import Adam
from pommerman.configs import ffa_competition_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.constants import BOARD_SIZE
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.core import Env, Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback
from rl.random import OrnsteinUhlenbeckProcess

tf.reset_default_graph()
# In[2]:


number_of_training_steps = 500000
log_interval = 10000
file_log_path = './dqn/rl_logs/ddgp_go_3blocks/log.txt'
tensorboard_path = './dqn/logs/ddgp_go_3blocks/'
model_path = './dqn/model/ddgp_go_3blocks/model{step}.h4'
PRETRAINED_MODEL_PATH = './dagger/model/il_go_3res_block/model.h4'


# In[3]:


if not os.path.isdir(os.path.dirname(file_log_path)):
    os.makedirs(os.path.dirname(file_log_path))
if not os.path.isdir(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))


# In[4]:


class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


class TensorboardLogger(Callback):
    """Logging in tensorboard without tensorflow ops."""
    def __init__(self, log_dir):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        episode_steps = len(self.observations[episode])
        variables = {
            'step': self.step,
            'episode_steps': episode_steps,
            'episode_reward': np.sum(self.rewards[episode]),
            'action_mean': np.mean(np.argmax(self.actions[episode], axis=1)),
            'action_min': np.min(np.argmax(self.actions[episode], axis=1)),
            'action_max': np.max(np.argmax(self.actions[episode], axis=1)),
        }

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    value = np.nanmean(metrics[:, idx])
                except Warning:
                    value = -1
                variables[name] = value
        for key, value in variables.items():
            self.log_scalar(key, value, episode + 1)

        # Free up resources.
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


# In[18]:


# Instantiate the environment
config = ffa_competition_env()
env = Pomme(**config["env_kwargs"])
np.random.seed(0)
env.seed(0)

env.set_init_game_state(None)
nb_actions = env.action_space.n


def get_res_block(input):
    # Res block 1
    x = Convolution2D(256, 3, padding='same')(input)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, padding='same')(x)
    #x = BatchNormalization()(x)
    x = Add()([input, x])
    x = Activation('relu')(x)
    return x


def create_actor(actions, input_shape=(BOARD_SIZE, BOARD_SIZE, 17,)):
    inp = Input(input_shape)
    x = Convolution2D(256, 3, padding='same')(inp)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Ten residual blocks
    for i in range(3):
        x = get_res_block(x)

    # Output block
    # Should be 2 filters
    x = Convolution2D(4, 1, padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    out = Dense(actions, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    #model.load_weights(PRETRAINED_MODEL_PATH)
    return model


def create_critic(actions, input_shape=(BOARD_SIZE, BOARD_SIZE, 17,)):
    action_input = Input(shape=(actions,), name='action_input')

    obs_inp = Input(input_shape)
    x = Convolution2D(256, 3, padding='same')(obs_inp)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Ten residual blocks
    for i in range(3):
        x = get_res_block(x)

    # Output block
    # Should be 2 filters
    x = Convolution2D(4, 1, padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    out = Dense(actions, activation='softmax')(x)
    model = Model(inputs=obs_inp, outputs=out)
    #model.load_weights(PRETRAINED_MODEL_PATH)

    x = Concatenate()([action_input, x])
    x = Dense(128, activation='elu')(x)
    out = Dense(1)(x)

    model = Model(inputs=[action_input, obs_inp], outputs=out)
    return action_input, model



actor = create_actor(nb_actions)
action_input, critic = create_critic(nb_actions)


# In[9]:


class EnvWrapper(Env):
    """The abstract environment class that is used by all agents. This class has the exact
        same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
        OpenAI Gym implementation, this class only defines the abstract methods without any actual
        implementation.
        To implement your own environment, you need to define the following methods:
        - `step`
        - `reset`
        - `render`
        - `close`
        Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
        """
    reward_range = (-1, 1)
    action_space = None
    observation_space = None

    def __init__(self, gym, board_size):
        self.gym = gym
        self.action_space = gym.action_space
        self.observation_space = gym.observation_space
        self.reward_range = gym.reward_range
        self.board_size = board_size

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        action = np.argmax(action)
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)
        agent_state = self.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, agent_reward, terminal, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        # Add 3 random agents
        train_agent_pos = np.random.randint(0, 4)
        agents = []
        for agent_id in range(4):
            if agent_id == train_agent_pos:
                agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
            else:
                agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
        self.gym.set_agents(agents)
        self.gym.set_training_agent(agents[train_agent_pos].agent_id)
        
        obs = self.gym.reset()
        agent_obs = self.featurize(obs[self.gym.training_agent])
        return agent_obs

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        self.gym.render(mode=mode, close=close)

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.gym.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise self.gym.seed(seed)

    def featurize(self, obs):
        shape = (BOARD_SIZE, BOARD_SIZE, 1)

        def get_matrix(dict, key):
            res = dict[key]
            return res.reshape(shape).astype(np.float32)

        def get_map(board, item):
            map = np.zeros(shape)
            map[board == item] = 1
            return map

        board = get_matrix(obs, 'board')

        # TODO: probably not needed Passage = 0
        rigid_map = get_map(board, 1)               # Rigid = 1
        wood_map = get_map(board, 2)                # Wood = 2
        bomb_map = get_map(board, 3)                # Bomb = 3
        flames_map = get_map(board, 4)              # Flames = 4
        fog_map = get_map(board, 5)                 # TODO: not used for first two stages Fog = 5
        extra_bomb_map = get_map(board, 6)          # ExtraBomb = 6
        incr_range_map = get_map(board, 7)          # IncrRange = 7
        kick_map = get_map(board, 8)                # Kick = 8
        skull_map = get_map(board, 9)               # Skull = 9

        position = obs["position"]
        my_position = np.zeros(shape)
        my_position[position[0], position[1], 0] = 1

        team_mates = get_map(board, obs["teammate"].value) # TODO during documentation it should be an array

        enemies = np.zeros(shape)
        for enemy in obs["enemies"]:
            enemies[board == enemy.value] = 1

        bomb_blast_strength = get_matrix(obs, 'bomb_blast_strength')
        bomb_life = get_matrix(obs, 'bomb_life')

        ammo = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["ammo"])
        blast_strength = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["blast_strength"])
        can_kick = np.full((BOARD_SIZE, BOARD_SIZE, 1), int(obs["can_kick"]))

        obs = np.concatenate([my_position, enemies, team_mates, rigid_map,
                              wood_map, bomb_map, flames_map,
                              fog_map, extra_bomb_map, incr_range_map,
                              kick_map, skull_map, bomb_blast_strength,
                              bomb_life, ammo, blast_strength, can_kick], axis=2)
        return obs 

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class CustomProcessor(Processor):
    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        # Arguments
            batch (list): List of states
        # Returns
            Processed list of states
        """
        batch = np.squeeze(batch, axis=1)
        return batch

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        info['result'] = info['result'].value
        return info


# In[19]:


env_wrapper = EnvWrapper(env, BOARD_SIZE)
processor = CustomProcessor()


memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=512, nb_steps_warmup_actor=512,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  batch_size=512, processor=processor)
agent.compile(Adam(lr=0.0001, clipnorm=1.), metrics=['mae'])


file_logger = FileLogger(file_log_path, interval=log_interval)
checkpoint = ModelIntervalCheckpoint(model_path, interval=log_interval)
tensorboard = TensorboardLogger(tensorboard_path)
callbacks=[file_logger, checkpoint, tensorboard]
if os.path.isfile(model_path):
    agent.load_weights(model_path)


# In[ ]:


history = agent.fit(env_wrapper, nb_steps=number_of_training_steps, visualize=False, verbose=2,
        nb_max_episode_steps=env._max_steps, callbacks=callbacks)


# In[ ]:


agent.save_weights(model_path, overwrite=True)


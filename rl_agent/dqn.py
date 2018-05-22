import numpy as np
import os
import tensorflow as tf
import warnings

from keras.models import Model
from keras.layers import Dense, Flatten, Convolution2D, Input
from keras.optimizers import Adam

from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.constants import BOARD_SIZE
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Env, Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback

number_of_training_steps = 5000000
log_interval = 1000
file_log_path = './dqn/rl_logs/cnn_128_3_3_dense_128_1/log.txt'
tensorboard_path = './dqn/logs/cnn_128_3_3_dense_128_1/'
model_path = './dqn/model/cnn_128_3_3_dense_128_1/model{step}.h4'

if not os.path.isdir(os.path.dirname(file_log_path)):
    os.makedirs(os.path.dirname(file_log_path))
if not os.path.isdir(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))


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
            'nb_steps': self.params['nb_steps'],
            'episode_steps': episode_steps,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
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


# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])
np.random.seed(0)
env.seed(0)
# Add 3 random agents
agents = []
for agent_id in range(3):
    agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent
agent_id += 1
agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
env.set_agents(agents)
env.set_training_agent(agents[-1].agent_id)
env.set_init_game_state(None)
nb_actions = env.action_space.n


def create_model(actions, input_shape=(13, 13, 17,)):
    inp = Input(input_shape)
    x = Convolution2D(128, 3, activation='relu')(inp)
    x = Convolution2D(128, 3, activation='relu')(x)
    x = Convolution2D(128, 3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(actions)(x)
    model = Model(inputs=inp, outputs=out)
    return model


# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
model = create_model(nb_actions)
print(model.summary())


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

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def featurize(self, obs):
        shape = (self.board_size, self.board_size, 1)

        def get_matrix(dict, key):
            res = dict[key]
            return res.reshape(shape).astype(np.float32)

        def get_map(board, item):
            map = np.zeros(shape)
            map[board == item] = 1
            return map

        board = get_matrix(obs, 'board')

        # TODO: probably not needed Passage = 0
        rigid_map = get_map(board, 1)  # Rigid = 1
        wood_map = get_map(board, 2)  # Wood = 2
        bomb_map = get_map(board, 3)  # Bomb = 3
        flames_map = get_map(board, 4)  # Flames = 4
        fog_map = get_map(board, 5)  # TODO: not used for first two stages Fog = 5
        extra_bomb_map = get_map(board, 6)  # ExtraBomb = 6
        incr_range_map = get_map(board, 7)  # IncrRange = 7
        kick_map = get_map(board, 8)  # Kick = 8
        skull_map = get_map(board, 9)  # Skull = 9

        position = obs["position"]
        my_position = np.zeros(shape)
        my_position[position[0], position[1], 0] = 1

        team_mates = get_map(board, obs["teammate"].value)  # TODO during documentation it should be an array

        enemies = np.zeros(shape)
        for enemy in obs["enemies"]:
            enemies[board == enemy.value] = 1

        bomb_blast_strength = get_matrix(obs, 'bomb_blast_strength')
        bomb_life = get_matrix(obs, 'bomb_life')

        ammo = np.full(shape, obs["ammo"])
        blast_strength = np.full(shape, obs["blast_strength"])
        can_kick = np.full(shape, int(obs["can_kick"]))

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


env_wrapper = EnvWrapper(env, BOARD_SIZE)
processor = CustomProcessor()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=1)
policy = BoltzmannQPolicy()
file_logger = FileLogger(file_log_path, interval=log_interval)
checkpoint = ModelIntervalCheckpoint(model_path, interval=log_interval)
tensorboard = TensorboardLogger(tensorboard_path)
callbacks=[file_logger, checkpoint, tensorboard]
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=30,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy,
               processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
if os.path.isfile(model_path):
    dqn.load_weights(model_path)
history = dqn.fit(env_wrapper, nb_steps=number_of_training_steps, visualize=False, verbose=2,
        nb_max_episode_steps=env._max_steps,
                  callbacks=callbacks)
dqn.save_weights(model_path, overwrite=True)
import argparse
import multiprocessing
from queue import Empty
import random
import os
import re
import glob
import numpy as np

from pommerman.constants import BOARD_SIZE
from pommerman.envs.v0 import Pomme
from pommerman.configs import ffa_competition_env
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
from tensorboard_utils import create_summary_writer, add_summary

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm

NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)


class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)
        assert p.shape == (NUM_ACTIONS,)
        self.P = p

    def action(self):
        U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        # TODO: use random tie-breaking for equal values
        return np.argmax(self.Q + U)

    def update(self, action, reward):
        self.W[action] += reward
        self.N[action] += 1
        self.Q[action] = self.W[action] / self.N[action]

    def probs(self, temperature=1):
        if temperature == 0:
            p = np.zeros(NUM_ACTIONS)
            p[np.argmax(self.N)] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt)


class MCTSAgent(BaseAgent):
    def __init__(self, model_file, agent_id=0):
        config = ffa_competition_env()
        super().__init__(config["agent"](agent_id, config["game_type"]))
        self.agent_id = agent_id
        self.env = self.make_env(config)
        self.model = load_model(model_file)
        self.reset_tree()

    # def make_env(self):
    #     agents = []
    #     for agent_id in range(NUM_AGENTS):
    #         if agent_id == self.agent_id:
    #             agents.append(self)
    #         else:
    #             agents.append(SimpleAgent())
    #
    #     return pommerman.make('PommeFFACompetition-v0', agents)

    def make_env(self, config):
        # Instantiate the environment
        env = Pomme(**config["env_kwargs"])
        # Add agents
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
        env.set_agents(agents)
        env.set_init_game_state(None)
        return env

    def reset_tree(self):
        self.tree = {}
        # for statistics
        self.hit_probs = []
        self.avg_lengths = []
        self.entropies = []

    def observation_to_features(self, obs):
        shape = (BOARD_SIZE, BOARD_SIZE, 1)

        def get_matrix(board, key):
            res = board[key]
            return res.reshape(shape).astype(np.float32)

        def get_map(board, item):
            map = np.zeros(shape)
            map[board == item] = 1
            return map

        board = get_matrix(obs, 'board')

        path_map = get_map(board, 0)  # Empty space
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

        ammo = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["ammo"])
        blast_strength = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["blast_strength"])
        can_kick = np.full((BOARD_SIZE, BOARD_SIZE, 1), int(obs["can_kick"]))

        obs = np.concatenate([my_position, enemies, team_mates, path_map, rigid_map,
                              wood_map, bomb_map, flames_map, fog_map, extra_bomb_map,
                              incr_range_map, kick_map, skull_map, bomb_blast_strength,
                              bomb_life, ammo, blast_strength, can_kick], axis=2)
        return obs.astype(np.int16)

    def search(self, root, num_iters, temperature=1):
        # remember current game state
        self.env._init_game_state = root
        root = str(self.env.get_json_info())

        # for statistics
        hits = 0
        misses = 0
        total_length = 0
        for i in range(num_iters):
            # restore game state to root node
            obs = self.env.reset()
            # serialize game state
            state = str(self.env.get_json_info())

            trace = []
            done = False
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    action = node.action()
                    trace.append((node, action))
                    hits += 1
                else:
                    # initialize action probabilities with policy network
                    feats = self.observation_to_features(obs[self.agent_id])
                    feats = np.array([feats])
                    probs, values = self.model.predict(feats)
                    probs = probs[0]
                    reward = values[0, 0]
                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)
                    misses += 1
                    break

                # ensure we are not called recursively
                assert self.env.training_agent == self.agent_id
                # make other agents act
                actions = self.env.act(obs)
                # add my action to list of actions
                actions.insert(self.agent_id, action)
                # step environment forward
                obs, rewards, done, info = self.env.step(actions)
                reward = rewards[self.agent_id]

                state = str(self.env.get_json_info())

            total_length += len(trace)

            # update tree nodes with rollout results
            for node, action in reversed(trace):
                node.update(action, reward)
                reward *= args.discount

        self.hit_probs.append(hits / (hits + misses))
        self.avg_lengths.append(total_length / num_iters)

        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None
        # return action probabilities
        pi = self.tree[root].probs(temperature)
        idx = (pi != 0)
        self.entropies.append(-np.sum(pi[idx] * np.log(pi[idx])))
        return pi

    def rollout(self, finished):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively and episode ends when this agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.reset()

        trace = []
        done = False
        while not done and not finished.value:
            # use temperature 1 for first 30 steps and temperature 0 afterwards
            # temp = 0 if self.env._step_count < 30 else 0
            # TODO: only works when agent has access to the env
            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, args.mcts_iters, args.temperature)
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)
            # record observations and action probabilities
            feats = self.observation_to_features(obs[self.agent_id])
            trace.append((feats, pi))

            # ensure we are not called recursively
            assert self.env.training_agent == self.agent_id
            # make other agents act
            actions = self.env.act(obs)
            # add my action to list of actions
            actions.insert(self.agent_id, action)
            # step environment
            obs, rewards, done, info = self.env.step(actions)
            assert self == self.env._agents[self.agent_id]

        reward = rewards[self.agent_id]
        return trace, reward

    def act(self, obs, action_space):
        obs = self.observation_to_features(obs)
        obs = np.array([obs])
        actions, reward = self.model.predict(obs)
        actions = actions[0]
        return np.argmax(actions)
        # sample action from probabilities
        # return np.random.choice(NUM_ACTIONS, p=pi)


class ReplayMemory(object):
    def __init__(self, size=100000):
        self.observations = np.empty((size, constants.BOARD_SIZE, constants.BOARD_SIZE, 18))
        self.action_probs = np.empty((size, NUM_ACTIONS))
        self.state_values = np.empty((size,))
        self.size = size
        self.current = 0
        self.count = 0

    def add_sample(self, obs, pi, z):
        self.observations[self.current] = obs
        self.action_probs[self.current] = pi
        self.state_values[self.current] = z
        self.current = (self.current + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def dataset(self):
        return self.observations[:self.count], self.action_probs[:self.count], self.state_values[:self.count]


def get_res_block(input):
    # Res block 1
    x = Convolution2D(256, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([input, x])
    x = Activation('relu')(x)
    return x


def make_model(actions=6, input_shape=(constants.BOARD_SIZE, constants.BOARD_SIZE, 18)):
    inp = Input(input_shape)
    x = Convolution2D(256, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3 residual blocks
    for i in range(3):
        x = get_res_block(x)

    # Output block
    # Should be 2 filters
    x = Convolution2D(2, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)

    probs = Dense(actions, activation='softmax', name='actions')(x)
    reward = Dense(1, name='reward')(x)

    model = Model(inputs=inp, outputs=[probs, reward])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mae'], metrics=['accuracy'])
    return model


def load_model(model_path):
    model = make_model()
    model.load_weights(model_path)
    return model


def init_tensorflow():
    # make sure TF does not allocate all memory
    # NB! this needs to be done also in subprocesses!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))


def get_model_path(args, logdir):
    files = glob.glob(os.path.join(logdir, "model_*.hdf5"))
    if files:
        model_file = max(files, key=lambda f: int(re.search(r'_(\d+).hdf5', f).group(1)))
        # set start timestep from file name when continuing previous session
        num_episodes = int(re.search(r'_(\d+).hdf5', model_file).group(1))
        print("Setting start episode to %d" % num_episodes)
        return model_file, num_episodes

    return args.load_model, 0


def runner(id, model_file, fifo, finished, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # initialize tensorflow
    init_tensorflow()
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(model_file, agent_id=agent_id)

    while not finished.value:
        # do rollout
        trace, reward = agent.rollout(finished)
        # don't put last trace into fifo
        if finished.value:
            break
        # add data samples to training set
        fifo.put((trace, reward, agent_id, agent.hit_probs, agent.avg_lengths, len(agent.tree), agent.entropies))
    print("RUNNER KILLED")


def trainer(num_iters, num_rollouts, model, writer, logdir):
    while num_iters < args.num_iters:
        print("################### ITERATION %d ###################" % (num_iters + 1))
        # Stats
        stat_tree_size = []
        stat_hit_probs = []
        stat_avg_lengths = []
        stat_entropies = []
        stat_reward_agent = [[], [], [], []]
        stat_episode_length = []

        memory = ReplayMemory()
        # -------Generate training set based on MCTS-------
        print("Generate dataset")
        # use spawn method for starting subprocesses
        # this seems to be more compatible with TensorFlow?
        ctx = multiprocessing.get_context('spawn')
        # create boolean to signal end
        finished = ctx.Value('i', 0)

        # create fifos and processes for all runners
        print("Creating child processes")
        fifos = []
        model_file, _ = get_model_path(args, logdir)
        for i in range(args.num_runners):
            fifo = ctx.Queue(1)
            fifos.append(fifo)
            process = ctx.Process(target=runner, args=(i, model_file, fifo, finished, args))
            process.start()

        for i in tqdm(range(num_rollouts)):
            while True:
                # pick random fifo (agent)
                fifo = random.choice(fifos)
                try:
                    # wait for a new trajectory and statistics
                    trace, reward, agent_id, hit_probs, avg_lengths, tree_size, entropies = fifo.get(timeout=1)
                    break
                except Empty:
                    pass
            # save stats
            stat_tree_size.append(tree_size)
            stat_hit_probs.append(np.mean(hit_probs))
            stat_avg_lengths.append(np.mean(avg_lengths))
            stat_entropies.append(np.mean(entropies))
            stat_reward_agent[agent_id].append(reward)
            stat_episode_length.append(len(trace))
            # add samples to replay memory
            for obs, pi in trace:
                memory.add_sample(obs, pi, reward)

        # Kill subprocesses
        finished.value = 1
        print("Finishing")
        # empty queues until all child processes have exited
        while len(multiprocessing.active_children()) > 0:
            for i, fifo in enumerate(fifos):
                if not fifo.empty():
                    fifo.get_nowait()
        print("All childs was killed")
        # -------Train a model-------
        callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto'),
                     ModelCheckpoint(os.path.join(logdir, "model_%d.hdf5" % num_iters), monitor='loss', save_best_only=True),
                     ReduceLROnPlateau(monitor='loss', patience=1, factor=0.1)]
        add_summary(writer, "tree/mean_size", np.mean(stat_tree_size), num_iters)
        try:
            add_summary(writer, "tree/mean_hit_prob", float(np.mean(stat_hit_probs)), num_iters)
        except:
            pass
        add_summary(writer, "tree/mean_rollout_len", float(np.mean(stat_avg_lengths)), num_iters)
        add_summary(writer, "episode/mean_entropy", float(np.mean(stat_entropies)), num_iters)
        try:
            add_summary(writer, "episode/reward", np.mean(stat_reward_agent), num_iters)
        except:
            pass
        add_summary(writer, "episode/length", np.mean(stat_episode_length), num_iters)
        add_summary(writer, "rewards/agent_id", agent_id, num_iters)
        for i in range(len(stat_reward_agent)):
            try:
                add_summary(writer, "rewards/agent%d" % i, np.mean(stat_reward_agent[i]), num_iters)
            except:
                pass

        X, y, z = memory.dataset()
        assert len(X) != 0

        # train for limited epochs to avoid overfitting?
        # TODO class weights??
        history = model.fit(X, [y, z], batch_size=args.batch_size, epochs=args.num_epochs, callbacks=callbacks,
                            validation_split=args.validation_split, shuffle=True)
        # log loss values
        for k, v in history.history.items():
            add_summary(writer, "training/" + k, v[-1], num_iters)
        num_iters += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--load_model')
    parser.add_argument('--logdir', default="logs")
    parser.add_argument('--num_rollouts', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=10)
    # runner params
    parser.add_argument('--num_runners', type=int, default=4)
    # trainer params
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--validation_split', type=float, default=0.15)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=10)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    logdir = os.path.join(args.logdir, args.label)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # check for commandline argument or previous model file
    model_file, num_iters = get_model_path(args, logdir)

    # load saved model
    init_tensorflow()
    if model_file:
        print("Loading model:", model_file)
        model = load_model(model_file)
    else:
        print("Initializing new model")
        model = make_model()
    model.summary()

    writer = create_summary_writer(logdir)

    # do training in main process
    print("Starting training in main process")
    trainer(num_iters, args.num_rollouts, model, writer, logdir)
    print("All done")

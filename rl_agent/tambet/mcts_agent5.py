import argparse
import multiprocessing
from queue import Empty
import random
import time
import os
import re
import glob
import pickle
import numpy as np
import time

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf


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
    def __init__(self, model_file=None, train=False, agent_id=0):
        super().__init__()
        self.agent_id = agent_id

        if train:
            self.env = self.make_env()

        if model_file is None:
            self.model = make_model()
        else:
            self.model = load_model(model_file)

        self.reset_tree()

    def make_env(self):
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())

        return pommerman.make('PommeFFACompetition-v0', agents)

    def reset_tree(self):
        self.tree = {}
        # for statistics
        self.hit_probs = []
        self.avg_lengths = []
        self.entropies = []
        self.iters_sec = []

    def observation_to_features(self, obs):
        # TODO: history of n moves?
        board = obs['board']

        # convert board items into bitmaps
        maps = [board == i for i in range(1, 10)]
        maps.append(obs['bomb_blast_strength'])
        maps.append(obs['bomb_life'])

        # duplicate ammo, blast_strength and can_kick over entire map
        maps.append(np.full(board.shape, obs['ammo']))
        maps.append(np.full(board.shape, obs['blast_strength']))
        maps.append(np.full(board.shape, obs['can_kick']))

        # add my position as bitmap
        position = np.zeros(board.shape)
        position[obs['position']] = 1
        maps.append(position)

        # add teammate
        if obs['teammate'] is not None:
            maps.append(board == obs['teammate'].value)
        else:
            maps.append(np.zeros(board.shape))

        # add enemies
        enemies = [board == e.value for e in obs['enemies']]
        maps.append(np.any(enemies, axis=0))

        return np.stack(maps, axis=2)

    def search(self, root, num_iters, temperature=1):
        # remember current game state
        self.env._init_game_state = root
        root = str(self.env.get_json_info())

        # for statistics
        hits = 0
        misses = 0
        total_length = 0
        start_time = time.time()
        for i in range(num_iters):
            # restore game state to root node
            obs = self.env.reset()
            print('\rStep %d: iteration %d' % (self.env._step_count, i + 1), end=' ')
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
                    #print("Action from tree:", constants.Action(action).name)
                    hits += 1
                else:
                    # initialize action probabilities with policy network
                    feats = self.observation_to_features(obs[self.agent_id])
                    feats = feats[np.newaxis, ...]
                    probs, values = self.model.predict(feats)
                    probs = probs[0]
                    reward = values[0, 0]

                    # add Dirichlet noise to root node for added exploration
                    # Hex people didn't find it necessary
                    #if len(trace) == 0:
                    #    noise = np.random.dirichlet([args.mcts_dirichlet_alpha] * len(probs))
                    #    probs = (1 - args.mcts_dirichlet_epsilon) * probs + args.mcts_dirichlet_epsilon * noise

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)

                    misses += 1
                    #print("Leaf node")
                    # stop at leaf node
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
                #print("Rewards:", rewards)

                state = str(self.env.get_json_info())

            total_length += len(trace)

            #print("Finished rollout, length:", len(trace))
            #print("Backpropagating reward:", reward)

            # update tree nodes with rollout results
            for node, action in reversed(trace):
                node.update(action, reward)
                reward *= args.discount

            #print("Root Q:")
            #print(self.tree[root].Q)
            #print("Root N:")
            #print(self.tree[root].N)
            print(self.tree[root].N, self.tree[root].Q, end='')

        #print("(tree hits: %0.2f, avg. len: %0.2f, tree size: %d)" % (hits / (hits + misses), total_length / num_iters, len(self.tree)))
        elapsed = time.time() - start_time
        self.iters_sec.append(num_iters / elapsed)
        self.hit_probs.append(hits / (hits + misses))
        self.avg_lengths.append(total_length / num_iters)

        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None
        # return action probabilities
        pi = self.tree[root].probs(temperature)
        print()
        idx = (pi != 0)
        self.entropies.append(-np.sum(pi[idx] * np.log(pi[idx])))
        return pi

    def rollout(self, shared_buffer, finished):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.reset()

        trace = []
        done = False
        while not done and not finished.value:
            if args.render:
                self.env.render()

            # copy weights from trainer
            self.model.set_weights(pickle.loads(shared_buffer.raw))

            # use temperature 1 for first 30 steps and temperature 0 afterwards
            #temp = 0 if self.env._step_count < 30 else 0
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
            print("Agent:", self.agent_id, "Step:", self.env._step_count, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Entropy: %.2f" % self.entropies[-1], "Iters/s: %.2f" % self.iters_sec[-1], "Rewards:", rewards, "Done:", done)

            #print("Rollout finished:", finished.value)

        reward = rewards[self.agent_id]
        #print("Agent:", self.agent_id, "Reward:", reward, "Len trace:", len(trace))
        return trace, reward, rewards

    def act(self, obs, action_space):
        obs = self.observation_to_features(obs)
        obs = np.array([obs])
        probs, reward = self.model.predict(obs)
        probs = probs[0]
        return np.argmax(probs)
        # sample action from probabilities
        #return np.random.choice(NUM_ACTIONS, p=pi)


class ReplayMemory(object):
    def __init__(self, size=100000):
        self.observations = np.empty((size, constants.BOARD_SIZE, constants.BOARD_SIZE, 17))
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


def make_model():
    c = x = Input(shape=(constants.BOARD_SIZE, constants.BOARD_SIZE, 17))
    for i in range(args.conv_layers):
        c = Conv2D(args.conv_filters, args.conv_filter_size, activation='relu', padding='valid')(c)
    h = Flatten()(c)
    for i in range(args.hidden_layers):
        h = Dense(args.hidden_nodes, activation='relu')(h)
    hp = h
    for i in range(args.policy_hidden_layers):
        hp = Dense(args.hidden_nodes, activation='relu')(hp)
    p = Dense(6, activation='softmax', kernel_initializer=RandomNormal(0.0, 0.001), name="policy")(hp)
    hv = h
    for i in range(args.value_hidden_layers):
        hv = Dense(args.hidden_nodes, activation='relu')(hv)
    v = Dense(1, activation='tanh', kernel_initializer=RandomNormal(0.0, 0.001), name="value")(hv)
    model = Model(x, [p, v])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'])
    return model

def init_tensorflow():
    # make sure TF does not allocate all memory
    # NB! this needs to be done also in subprocesses!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

def runner(id, model_file, shared_buffer, fifo, finished, _args):
    import sys
    sys.stdin = open("/dev/stdin", "r")
    # make args accessible to MCTSAgent
    global args
    args = _args
    # initialize tensorflow
    init_tensorflow()
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(model_file, train=True, agent_id=agent_id)

    while not finished.value:
        # do rollout
        trace, reward, rewards = agent.rollout(shared_buffer, finished)
        # don't put last trace into fifo
        if finished.value:
            break
        # add data samples to training set
        fifo.put((trace, reward, rewards, agent_id, agent.hit_probs, agent.avg_lengths, len(agent.tree), agent.entropies, agent.iters_sec))
        #print("Runner finished:", finished.value)

    #print("Runner done")

def trainer(num_episodes, fifos, shared_buffer, model, memory, writer):
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')]
    while num_episodes < args.num_episodes:
        while True:
            # pick random fifo (agent)
            fifo = random.choice(fifos)
            try:
                # wait for a new trajectory and statistics
                trace, reward, rewards, agent_id, hit_probs, avg_lengths, tree_size, entropies, iters_sec = fifo.get(timeout=args.queue_timeout)
                # break out of the infinite loop
                break
            except Empty:
                # just ignore empty fifos
                pass

        num_episodes += 1

        # add samples to replay memory
        # TODO: add_batch would be more efficient?
        for obs, pi in trace:
            memory.add_sample(obs, pi, reward)

        add_summary(writer, "tree/size", tree_size, num_episodes)
        add_summary(writer, "tree/mean_hit_prob", float(np.mean(hit_probs)), num_episodes)
        add_summary(writer, "tree/mean_rollout_len", float(np.mean(avg_lengths)), num_episodes)
        add_summary(writer, "tree/iters_sec", float(np.mean(iters_sec)), num_episodes)
        add_histogram(writer, "tree/hit_probability", hit_probs, num_episodes)
        add_histogram(writer, "tree/rollout_length", avg_lengths, num_episodes)
        add_histogram(writer, "tree/entropies", entropies, num_episodes)
        add_summary(writer, "episode/mean_entropy", float(np.mean(entropies)), num_episodes)
        add_summary(writer, "episode/reward", reward, num_episodes)
        add_summary(writer, "episode/length", len(trace), num_episodes)
        add_summary(writer, "rewards/agent_id", agent_id, num_episodes)
        for i in range(len(rewards)):
            add_summary(writer, "rewards/agent%d" % i, rewards[i], num_episodes)
        add_summary(writer, "replay_memory/size", memory.size, num_episodes)
        add_summary(writer, "replay_memory/count", memory.count, num_episodes)
        add_summary(writer, "replay_memory/current", memory.current, num_episodes)

        #print("Replay memory size: %d, count: %d, current: %d" % (memory.size, memory.count, memory.current))
        X, y, z = memory.dataset()
        assert len(X) != 0

        # reset weights?
        if args.reset_network:
            #model.set_weights(init_weights)
            model = model_from_json(model.to_json())
            model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'])
        # train for limited epochs to avoid overfitting?
        history = model.fit(X, [y, z], batch_size=args.batch_size, epochs=args.num_epochs, callbacks=callbacks, validation_split=args.validation_split)
        # log loss values
        for k, v in history.history.items():
            add_summary(writer, "training/" + k, v[-1], num_episodes)
        # shared weights with runners
        shared_buffer.raw = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)
        # save weights
        if num_episodes % args.save_interval == 0:
            model.save(os.path.join(logdir, "model_%d.hdf5" % num_episodes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--load_model')
    parser.add_argument('--logdir', default="logs")
    parser.add_argument('--render', action="store_true", default=False)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=10)
    # queue params
    parser.add_argument('--queue_length', type=int, default=10)
    parser.add_argument('--queue_timeout', type=int, default=1)
    # runner params
    parser.add_argument('--num_runners', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=constants.MAX_STEPS)
    # trainer params
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--reset_network', action='store_true', default=False)
    # network params
    parser.add_argument('--conv_layers', type=int, default=0)
    parser.add_argument('--conv_filters', type=int, default=32)
    parser.add_argument('--conv_filter_size', type=int, default=3)
    parser.add_argument('--hidden_layers', type=int, default=0)
    parser.add_argument('--hidden_nodes', type=int, default=32)
    parser.add_argument('--policy_hidden_layers', type=int, default=0)
    parser.add_argument('--value_hidden_layers', type=int, default=0)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=10)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    parser.add_argument('--mcts_dirichlet_epsilon', type=float, default=0.25)
    parser.add_argument('--mcts_dirichlet_alpha', type=float, default=0.3)
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    memory = ReplayMemory()
    logdir = os.path.join(args.logdir, args.label)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # use spawn method for starting subprocesses
    # this seems to be more compatible with TensorFlow?
    ctx = multiprocessing.get_context('spawn')

    # check for commandline argument or previous model file
    num_episodes = 0
    model_file = None
    if args.load_model:
        model_file = args.load_model
    else:
        files = glob.glob(os.path.join(logdir, "model_*.hdf5"))
        if files:
            model_file = max(files, key=lambda f: int(re.search(r'_(\d+).hdf5', f).group(1)))
            # set start timestep from file name when continuing previous session
            num_episodes = int(re.search(r'_(\d+).hdf5', model_file).group(1))
            print("Setting start episode to %d" % num_episodes)

    # load saved model
    init_tensorflow()
    if model_file:
        print("Loading model:", model_file)
        model = load_model(model_file)
    else:
        print("Initializing new model")
        model = make_model()
    model.summary()

    # create shared buffer for sharing weights
    print("Creating shared memory for model")
    init_weights = model.get_weights()
    blob = pickle.dumps(init_weights, pickle.HIGHEST_PROTOCOL)
    shared_buffer = ctx.Array('c', len(blob))
    shared_buffer.raw = blob

    # create boolean to signal end
    finished = ctx.Value('i', 0)

    # create fifos and processes for all runners
    print("Creating child processes")
    fifos = []
    for i in range(args.num_runners):
        fifo = ctx.Queue(args.queue_length)
        fifos.append(fifo)
        process = ctx.Process(target=runner, args=(i, model_file, shared_buffer, fifo, finished, args))
        process.start()

    from tensorboard_utils import create_summary_writer, add_summary, add_histogram
    writer = create_summary_writer(logdir)

    # do training in main process
    print("Starting training in main process")
    trainer(num_episodes, fifos, shared_buffer, model, memory, writer)
    finished.value = 1
    print("Finishing")
    # empty queues until all child processes have exited
    while len(multiprocessing.active_children()) > 0:
        for i, fifo in enumerate(fifos):
            if not fifo.empty():
                fifo.get_nowait()
    print("All done")

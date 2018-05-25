
# coding: utf-8

# In[1]:


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

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
from pommerman.constants import BOARD_SIZE

from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.callbacks import EarlyStopping, TensorBoard
import keras
import keras.backend as K
import tensorflow as tf


# In[2]:


NUM_AGENTS = 16
NUM_ACTIONS = len(constants.Action)

# environmental parameters
LOAD_MODEL = './dagger/model/il_go_3res_block/model.h4'
LOGDIR = './mcts/pretrained_go_3blocks/'
RENDER = False
NUM_EPISODES = 1000000 
SAVE_INTERVAL = 20 

# queue params
QUEUE_LENGTH = 10
QUEUE_TIMEOUT = 1

# runner params
NUM_RUNNERS = 16
MAX_STEPS = constants.MAX_STEPS

# trainer params
BATCH_SIZE = 2048
NUM_EPOCHS = 100
RESET_NETWORK = False

# MCTS params
MCTS_C_PUCT            = 1.0
MCTS_DIRICHLET_EPSILON = 0.25 
MCTS_ITERS             = 300
MCTS_DIRICHLET_ALPHA   = 0.3


# In[3]:


class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS)
        assert p.shape == (NUM_ACTIONS,)
        self.P = p

    def action(self):
        U = MCTS_C_PUCT * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
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


# In[4]:


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

    def observation_to_features(self, obs):
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
            #print('\rStep %d: iteration %d' % (self.env._step_count, i + 1), end=' ')
            # serialize game state
            state = str(self.env.get_json_info())

            trace = []
            done = False
            while not done:
                #print(board)
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
                    values = values[0]

                    # add Dirichlet noise to root node for added exploration
                    if len(trace) == 0:
                        noise = np.random.dirichlet([MCTS_DIRICHLET_ALPHA] * len(probs))
                        probs = (1 - MCTS_DIRICHLET_EPSILON) * probs + MCTS_DIRICHLET_EPSILON * noise

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)

                    # get a reward
                    rewards = self.env._get_rewards()
                    reward = rewards[self.agent_id]

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

                state = str(self.env.get_json_info())

            total_length += len(trace)

            #print("Finished rollout, length:", len(trace))
            #print("Backpropagating rewards:", rewards)

            # update tree nodes with rollout results
            for node, action in trace:
                node.update(action, reward)

            #print("Root Q:")
            #print(self.tree[root].Q)
            #print("Root N:")
            #print(self.tree[root].N)

        #print("(tree hits: %0.2f, avg. len: %0.2f, tree size: %d)" % (hits / (hits + misses), total_length / num_iters, len(self.tree)))
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
            if RENDER:
                self.env.render()

            # copy weights from trainer
            self.model.set_weights(pickle.loads(shared_buffer.raw))

            # use temperature 1 for first 30 steps and temperature 0 afterwards
            temp = 1 #if self.env._step_count < 30 else 0
            # TODO: only works when agent has access to the env
            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, MCTS_ITERS, temp)
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
            print("Agent:", self.agent_id, "Step:", self.env._step_count, "Actions:", [constants.Action(a).name for a in actions], "Entropy: %.2f" % self.entropies[-1], "Rewards:", rewards, "Done:", done)

            #print("Rollout finished:", finished.value)

        reward = rewards[self.agent_id]
        #print("Agent:", self.agent_id, "Reward:", reward, "Len trace:", len(trace))
        return trace, reward, rewards

    def act(self, obs, action_space):
        feats = self.observation_to_features(obs)
        feats = feats[np.newaxis, ...]
        probs, values = self.model.predict(feats)
        probs = probs[0]
        return np.argmax(probs)
        # sample action from probabilities
        #return np.random.choice(NUM_ACTIONS, p=pi)


# In[5]:


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


# In[6]:


def get_res_block(self, input):
    # Res block 1        
    x = Convolution2D(256, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([input, x])
    x = Activation('relu')(x)
    return x

def make_model():
    inp = Input(input_shape)
    x = Convolution2D(256, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Ten residual blocks
    for i in range(3):
        x = self.get_res_block(x)

    # Output block
    # Should be 2 filters
    x = Convolution2D(4, 1, padding='same')(x)
    x = BatchNormalization()(x)   
    x = Activation('relu')(x)
    x = Flatten()(x)
    out = Dense(actions, activation='softmax')(x)
    model = Model(inputs = inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_model(model_path):
    model = keras.models.load_model(LOAD_MODEL)
    inp = model.layers[0]
    probabilities = model.layers[-1].output
    h = model.layers[-2]
    reward = Dense(128, activation='relu', name='reward_1')(h.output)
    reward = Dense(1, activation='tanh', name='reward_out')(reward)
    model = Model(inp.input, [probabilities, reward])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'])
    return model

def init_tensorflow():
    # make sure TF does not allocate all memory
    # NB! this needs to be done also in subprocesses!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))


# In[7]:


def runner(id, model_file, shared_buffer, fifo, finished):
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
        fifo.put((trace, reward, rewards, agent_id, agent.hit_probs, agent.avg_lengths, len(agent.tree), agent.entropies))
        #print("Runner finished:", finished.value)

    #print("Runner done")


# In[8]:


def trainer(num_episodes, fifos, shared_buffer, model, memory, writer):
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.003, patience=5, verbose=1, mode='auto')
    tensorboard = TensorBoard(os.path.join(LOGDIR, 'tensorboard'), BATCH_SIZE)
    callbacks = [early_stopping, tensorboard]
    while num_episodes < NUM_EPISODES:
        while True:
            # pick random fifo (agent)
            fifo = random.choice(fifos)
            try:
                # wait for a new trajectory and statistics
                trace, reward, rewards, agent_id, hit_probs, avg_lengths, tree_size, entropies =                     fifo.get(timeout=QUEUE_TIMEOUT)
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
        if RESET_NETWORK:
            #model.set_weights(init_weights)
            model = model_from_json(model.to_json())
            model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'])
        # train for limited epochs to avoid overfitting?
        history = model.fit(X, [y, z], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=callbacks)
        # log loss values
        for k, v in history.history.items():
            add_summary(writer, "training/" + k, v[-1], num_episodes)
        # shared weights with runners
        shared_buffer.raw = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)
        # save weights
        if num_episodes % SAVE_INTERVAL == 0:
            model.save(os.path.join(LOGDIR, "model_%d.hdf5" % num_episodes))


# In[ ]:


if __name__ == "__main__":    
    global runner
    memory = ReplayMemory()
    if not os.path.isdir(LOGDIR):
        os.makedirs(LOGDIR)

    # use spawn method for starting subprocesses
    # this seems to be more compatible with TensorFlow?
    ctx = multiprocessing.get_context('spawn')

    # check for commandline argument or previous model file
    num_episodes = 0
    model_file = None
    if LOAD_MODEL:
        model_file = LOAD_MODEL
    else:
        files = glob.glob(os.path.join(LOGDIR, "model_*.hdf5"))
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
    for i in range(NUM_RUNNERS):
        fifo = ctx.Queue(QUEUE_LENGTH)
        fifos.append(fifo)
        process = ctx.Process(target=runner, args=(i, model_file, shared_buffer, fifo, finished))
        process.start()

    from tensorboard_utils import create_summary_writer, add_summary, add_histogram
    writer = create_summary_writer(LOGDIR)

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


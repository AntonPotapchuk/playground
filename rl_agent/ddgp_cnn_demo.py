import numpy as np
import os
import tensorflow as tf
import warnings

from keras.models import Model
from keras.layers import Dense, Flatten, Convolution2D, Input, Concatenate, Activation
from keras.optimizers import Adam

from pommerman.characters import Bomber
from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.constants import BOARD_SIZE


model_path = './dqn/model/ddgp_cnn128_3_3_dense_64_1/model.h4'
model_path2 = './dqn/model/ddgp_dense_8_2/model.h4'
model_path3 = './dqn/model/ddgp_dense_128_1/model.h4'

class Cnn12833Dense641(BaseAgent):
    def __init__(self, actions, board_size, save_path="./dagger/model/dense1/model.h4", character=Bomber):
        super(Cnn12833Dense1281, self).__init__(character=character)
        self.save_path = save_path
        self.board_size = board_size
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except Exception as ex:
                print("Model load failed", ex)
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(13, 13, 17,)):
        inp = Input(input_shape)
        x = Convolution2D(128, 3, activation='relu')(inp)
        x = Convolution2D(128, 3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(actions)(x)
        out = Activation('softmax')(out)
        model = Model(inputs=inp, outputs=out)
        return model

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

    def act(self, obs, action_space=None):
        obs = self.featurize(obs)
        obs = np.array([obs])
        predictions = self.model.predict(obs)
        print(np.argmax(predictions))
        print(predictions)
        return np.argmax(predictions)

class Cnn12832Dense1281(BaseAgent):
    def __init__(self, actions, board_size, save_path="./dagger/model/dense1/model.h4", character=Bomber):
        super(Cnn12832Dense1281, self).__init__(character=character)
        self.save_path = save_path
        self.board_size = board_size
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except Exception as ex:
                print("Model load failed", ex)
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(13, 13, 17,)):
        inp = Input(input_shape)
        x = Convolution2D(128, 3, activation='relu')(inp)
        x = Convolution2D(128, 3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(actions)(x)
        out = Activation('softmax')(out)
        model = Model(inputs=inp, outputs=out)
        return model

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

    def act(self, obs, action_space=None):
        obs = self.featurize(obs)
        obs = np.array([obs])
        predictions = self.model.predict(obs)
        print(np.argmax(predictions))
        print(predictions)
        return np.argmax(predictions)

class Dense82(BaseAgent):
    def __init__(self, actions, board_size, save_path="./dagger/model/dense1/model.h4", character=Bomber):
        super(Dense82, self).__init__(character=character)
        self.save_path = save_path
        self.board_size = board_size
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except Exception as ex:
                print("Model load failed", ex)
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(2369,)):
        inp = Input(input_shape)
        x = Dense(8)(inp)
        x = Activation('relu')(x)
        x = Dense(8)(x)
        x = Activation('relu')(x)
        out = Dense(actions)(x)
        out = Activation('softmax')(out)
        model = Model(inputs=inp, outputs=out)
        return model

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

        ammo = obs["ammo"]
        blast_strength = obs["blast_strength"]
        can_kick = int(obs["can_kick"])

        obs = np.concatenate([my_position, enemies, team_mates, rigid_map,
                              wood_map, bomb_map, flames_map,
                              fog_map, extra_bomb_map, incr_range_map,
                              kick_map, skull_map, bomb_blast_strength,
                              bomb_life], axis=2).flatten()
        obs = np.append(obs, [ammo, blast_strength, can_kick])
        return obs

    def act(self, obs, action_space=None):
        obs = self.featurize(obs)
        obs = np.array([obs])
        predictions = self.model.predict(obs)
        print(np.argmax(predictions))
        print(predictions)
        return np.argmax(predictions)

class Dense128(BaseAgent):
    def __init__(self, actions, board_size, save_path="./dagger/model/dense1/model.h4", character=Bomber):
        super(Dense128, self).__init__(character=character)
        self.save_path = save_path
        self.board_size = board_size
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except Exception as ex:
                print("Model load failed", ex)
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(2369,)):
        inp = Input(input_shape)
        x = Dense(128)(inp)
        x = Activation('relu')(x)
        out = Dense(actions)(x)
        out = Activation('softmax')(out)
        model = Model(inputs=inp, outputs=out)
        return model

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

        ammo = obs["ammo"]
        blast_strength = obs["blast_strength"]
        can_kick = int(obs["can_kick"])

        obs = np.concatenate([my_position, enemies, team_mates, rigid_map,
                              wood_map, bomb_map, flames_map,
                              fog_map, extra_bomb_map, incr_range_map,
                              kick_map, skull_map, bomb_blast_strength,
                              bomb_life], axis=2).flatten()
        obs = np.append(obs, [ammo, blast_strength, can_kick])
        return obs

    def act(self, obs, action_space=None):
        obs = self.featurize(obs)
        obs = np.array([obs])
        predictions = self.model.predict(obs)
        print(np.argmax(predictions))
        print(predictions)
        return np.argmax(predictions)

# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)

agent_pos = 2
# Add 3 random agents
agents = []
for agent_id in range(4):
    if agent_id == agent_pos:
        # agents.append(Cnn12833Dense1281(env.action_space.n, BOARD_SIZE, character=config["agent"](agent_id, config["game_type"]),
        #                       save_path=model_path))
        # agents.append(Dense82(env.action_space.n, BOARD_SIZE, character=config["agent"](agent_id, config["game_type"]),
        #                       save_path=model_path2))
        # agents.append(Dense128(env.action_space.n, BOARD_SIZE, character=config["agent"](agent_id, config["game_type"]),
        #                       save_path=model_path3))
        # agents.append(Dense128(env.action_space.n, BOARD_SIZE, character=config["agent"](agent_id, config["game_type"]),
        #                        save_path='./dqn/model/ddgp_dense_128_1_rs/model.h4'))
        agents.append(Cnn12832Dense1281(env.action_space.n, BOARD_SIZE, character=config["agent"](agent_id, config["game_type"]),
                              save_path='./dqn/model/ddgp_cnn128_3_2_dense_128_1_rs/model.h4'))
    else:
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

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

from sklearn.ensemble import BaggingClassifier

BaggingClassifier()


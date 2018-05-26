import os
import numpy as np

from keras.layers import Convolution2D, Input, Dense, Flatten, BatchNormalization, Activation, Add
from keras.models import Model
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.characters import Bomber
from pommerman.configs import ffa_competition_env
from pommerman.constants import BOARD_SIZE
from pommerman.envs.v0 import Pomme


save_path = "./dagger/model/cnn2dense1/model.h4"
save_dense2_path = "./dagger/model/dense1/model.h4"

class IL_cnn2_128_dense1(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/il_cnn2_128_dense1/model.h4", character=Bomber):
        super(IL_cnn2_128_dense1, self).__init__(character=character)
        self.save_path = save_path
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(13, 13, 17,)):
        inp = Input(input_shape)
        x = Convolution2D(128, 3)(inp)
        x = Convolution2D(128, 3)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(actions)(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        return np.argmax(predictions)

class Cnn3Dense1Trained(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/cnn3dense1/model.h4", character=Bomber):
        super(Cnn3Dense1Trained, self).__init__(character=character)
        self.save_path = save_path
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(13, 13, 17,)):
        inp = Input(input_shape)
        x = Convolution2D(64, 3)(inp)
        x = Convolution2D(64, 3)(x)
        x = Convolution2D(64, 3)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(actions)(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        return np.argmax(predictions)

class IL_dense1(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/il_dense1/model.h4", character=Bomber):
        super(IL_dense1, self).__init__(character=character)
        self.save_path = save_path
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
        x = Dense(128, activation='relu')(inp)
        out = Dense(actions)(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        return np.argmax(predictions)

class Dense1(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/dense1/model.h4", character=Bomber):
        super(Dense1, self).__init__(character=character)
        self.save_path = save_path
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
        x = Dense(128, activation='relu')(inp)
        out = Dense(actions)(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        return np.argmax(predictions)


class Il_go_1res_block(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/il_go_1res_block/model.h4", character=Bomber):
        super(Il_go_1res_block, self).__init__(character=character)
        self.save_path = save_path
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")
                raise Exception("Model load failed")

    def create_model(self, actions, input_shape=(11, 11, 17,)):
        inp = Input(input_shape)
        x = Convolution2D(256, 3, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Res block 1
        inp1 = x
        x = Convolution2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([inp1, x])
        x = Activation('relu')(x)

        # Output block
        # Should be 2 filters
        x = Convolution2D(4, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        out = Dense(actions, activation='softmax')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        print(predictions)
        return np.argmax(predictions)



    # Instantiate the environment


class Il_go_3res_block(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/il_go_3res_block/model.h4", character=Bomber):
        super(Il_go_3res_block, self).__init__(character=character)
        self.save_path = save_path
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")
                raise Exception("Model load failed")

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

    def create_model(self, actions, input_shape=(11, 11, 17,)):
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
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        print(predictions, np.argmax(predictions))
        return np.argmax(predictions)


class Il_go_5res_block(BaseAgent):
    def __init__(self, actions, save_path="./dagger/model/il_go_5res_block/model.h4", character=Bomber):
        super(Il_go_5res_block, self).__init__(character=character)
        self.save_path = save_path
        self.actions = actions

        self.model = self.create_model(actions)
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")
                raise Exception("Model load failed")

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

    def create_model(self, actions, input_shape=(11, 11, 17,)):
        inp = Input(input_shape)
        x = Convolution2D(256, 3, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Ten residual blocks
        for i in range(5):
            x = self.get_res_block(x)

        # Output block
        # Should be 2 filters
        x = Convolution2D(4, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        out = Dense(actions, activation='softmax')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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
        print(predictions, np.argmax(predictions))
        return np.argmax(predictions)


config = ffa_competition_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)

# Add 3 random agents
agents = []
train_id = 2
for agent_id in range(4):
    if train_id == agent_id:
        print("MY agent")
        agents.append(Il_go_3res_block(env.action_space.n, character=config["agent"](agent_id, config["game_type"])))
    else:
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent

#agents.append(Cnn3Dense1Trained(env.action_space.n, character=config["agent"](agent_id, config["game_type"])))
#agents.append(IL_dense1(env.action_space.n, character=config["agent"](agent_id, config["game_type"])))
#agents.append(IL_cnn2_128_dense1(env.action_space.n, character=config["agent"](agent_id, config["game_type"])))
#agents.append(Il_go_1res_block(env.action_space.n, character=config["agent"](agent_id, config["game_type"])))

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
import os
import numpy as np
import keras.backend as K

from keras.layers import Convolution2D, Input, Dense, Flatten, BatchNormalization, Activation, Add
from keras.models import Model
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.configs import ffa_competition_env
from pommerman.constants import BOARD_SIZE
from pommerman.envs.v0 import Pomme


model_path = './supervised_learning/model/go_3res_block/model.h4'


class Go_3resblock(BaseAgent):
    def __init__(self, actions, save_path, character):
        super(Go_3resblock, self).__init__(character)
        K.clear_session()
        self.save_path = save_path
        self.actions = actions

        # Create model
        self.model = self.create_model(actions)
        # Load model if exists
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if os.path.isfile(self.save_path):
            try:
                print("Trying to load model")
                self.model.load_weights(self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")

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

    def create_model(self, actions, input_shape=(11, 11, 18,)):
        inp = Input(input_shape)
        x = Convolution2D(256, 3, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 3 residual blocks
        for i in range(3):
            x = self.get_res_block(x)

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

    def featurize(self, obs):
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

    def act(self, obs, action_space=None):
        obs = self.featurize(obs)
        obs = np.array([obs])
        action, reward = self.model.predict(obs)
        print(action)
        return np.argmax(action)

config = ffa_competition_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)

# Add 3 random agents
agents = []
train_id = 2
for agent_id in range(4):
    if train_id == agent_id:
        agents.append(Go_3resblock(env.action_space.n, model_path, config["agent"](agent_id, config["game_type"])))
    else:
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

env.set_init_game_state(None)
env.set_agents(agents)

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
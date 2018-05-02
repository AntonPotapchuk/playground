import numpy as np

from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.configs import ffa_v0_env
from pommerman.constants import BOARD_SIZE, GameType
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

episodes = 100


def reward_function(agents, game_type, step_count, max_steps):
    def any_lst_equal(lst, values):
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) if agent.is_alive]
    if game_type == GameType.FFA:
        if len(alive_agents) == 1:
            # An agent won. Give them +1, others -1.
            return [2 * int(agent.is_alive) - 1 for agent in agents]
        elif step_count >= max_steps:
            # Game is over from time. Everyone gets -1.
            return [-1] * 4
        # TODO
        else:
            # Game running: 0 for alive, -1 for dead.
            return [int(agent.is_alive) - 1 for agent in agents]
    else:
        # We are playing a team game.
        if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
            # Team [0, 2] wins.
            return [1, -1, 1, -1]
        elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
            # Team [1, 3] wins.
            return [-1, 1, -1, 1]
        elif step_count >= max_steps:
            # Game is over by max_steps. All agents tie.
            return [-1] * 4
        # No team has yet won or lost.
        return [0] * 4


# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"], custom_reward=reward_function)
env.seed(0)

# Create a Proximal Policy Optimization agent
network = dict(type='rl_agent.pomm_network.PommNetwork')
states = {
    "board": dict(shape=(BOARD_SIZE, BOARD_SIZE, 3, ), type='float'),
    "state": dict(shape=(3,), type='float')
}
agent = PPOAgent(
    states=states,
    actions=dict(type='int', num_actions=env.action_space.n),
    network=network,
    batching_capacity=10000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass
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


class WrappedEnv(OpenAIGym):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = WrappedEnv.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = WrappedEnv.featurize(obs[3])
        return agent_obs

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


def episode_finished(r):
    if r.episode % 10 == 0:
        print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts = r.timestep + 1))
        print("Episode reward: {}".format(r.episode_rewards[-1]))
        print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[10:])))
    return True

# Instantiate and run the environment for 5 episodes.
wrapped_env = WrappedEnv(env, False)
runner = Runner(agent=agent, environment=wrapped_env)
runner.run(episodes=episodes, episode_finished=episode_finished)
print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)

try:
    runner.close()
except AttributeError as e:
    pass
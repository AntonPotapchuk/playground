import os
import shutil
import pommerman
import numpy as np
import tensorflow as tf

from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.configs import ffa_v0_env
from pommerman.constants import BOARD_SIZE
from pommerman.envs.v0 import Pomme
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.utils import to_categorical

# Immitation Learning: learn a mapping from observations to actions.
from rl_agent.pomm_network import PommNetwork

batch_size = 1024
epochs = 10
num_rollouts = 3

class Agent:
    def __init__(self, actions, seed=0, save_path="./model/model.cptk", log_path='./logs/', save_best_model=True, learning_rate=1e-3):
        self.save_path = save_path
        self.actions = actions
        self.save_best_model = save_best_model
        self.seed = seed
        self.sess = tf.InteractiveSession()
        self.rewards = []
        if os.path.isdir(log_path):
            try:
                shutil.rmtree(log_path, ignore_errors=True)
            except:
                print("Cant delete log folder")

        # TODO hardcoded
        self.conv_ph = tf.placeholder(shape=[None, BOARD_SIZE, BOARD_SIZE, 3], name='conv_ph', dtype=tf.float32)
        self.state_ph = tf.placeholder(shape=[None, 3], name='state_ph', dtype=tf.float32)
        self.logits_ph = tf.placeholder(shape=[None, actions], name='logits_ph', dtype=tf.int32)

        network = PommNetwork.create_network({'board': self.conv_ph, 'state': self.state_ph})
        logits = tf.layers.dense(network, actions)
        self.sampled_action = tf.squeeze(tf.multinomial(logits, 1), axis=[1])
        sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits(labels=self.logits_ph, logits=logits)
        self.loss = tf.reduce_mean(sy_logprob_n)  # Loss function that we'll differentiate to get the policy gradient.
        self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.train_writer = tf.summary.FileWriter(log_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(log_path + "test")
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if os.path.isdir(os.path.dirname(self.save_path)):
            try:
                print("Trying to load model")
                self.saver.restore(self.sess, self.save_path)
                print("Model was loaded successful")
            except:
                print("Model load failed")

    def __run_batch(self, X, y, batch_size, training=True):
        accuracies = []
        losses = []
        size = X.shape[-1]
        batches = int(np.ceil(size / batch_size))
        for i in range(batches):
            X_batch, y_batch = None, None
            if i == batches - 1:
                X_batch = X[i * batch_size:]
                y_batch = y[i * batch_size:]
            else:
                X_batch = X[i * batch_size: (i + 1) * batch_size]
                y_batch = y[i * batch_size: (i + 1) * batch_size]
            board = []
            state = []
            for i in range(X_batch.shape[-1]):
                val = self.featurize(X_batch[i])
                board.append(val['board'])
                state.append(val['state'])
            y_batch = np.array(y_batch).reshape(-1)
            feed_dict = {self.conv_ph: board, self.state_ph: state, self.logits_ph: to_categorical(y_batch, self.actions)}
            if training:
                _, loss, actions = self.sess.run([self.train_step, self.loss, self.sampled_action], feed_dict=feed_dict)
            else:
                loss, actions = self.sess.run([self.loss, self.sampled_action], feed_dict=feed_dict)
            accuracies.append(accuracy_score(actions, y_batch))
            losses.append(loss)
        return np.mean(accuracies), np.mean(losses)

    def train(self, obs, labels, batch_size, epochs):
        print("Train the agent with %i training data, batch_size %i, epochs %i" % (obs.shape[0], batch_size, epochs))
        train_obs, val_obs, train_labels, val_labels = train_test_split(obs, labels, test_size=0.2, random_state=self.seed)
        prev_loss = np.inf
        for i in range(epochs):
            train_acc, train_loss = self.__run_batch(train_obs, train_labels, batch_size)
            val_acc, val_loss = self.__run_batch(val_obs, val_labels, batch_size, training=False)
            print("Epoch %d: train_acc %f, train_loss %f, test_acc %f, test_loss: %f" % (i, train_acc, train_loss, val_acc, val_loss))

            summary = tf.summary.Summary()
            summary.value.add(tag="Train acc", simple_value=train_acc)
            summary.value.add(tag="Train loss", simple_value=train_loss)
            self.train_writer.add_summary(summary, i)
            summary.value.add(tag="Val acc", simple_value=val_acc)
            summary.value.add(tag="Val loss", simple_value=val_loss)
            self.test_writer.add_summary(summary, i)
            try:
                if self.save_best_model:
                    if val_loss < prev_loss:
                        print("Saving model")
                        self.saver.save(self.sess, self.save_path)
                        print("Model was saved successfully")
                else:
                    print("Saving model")
                    self.saver.save(self.sess, self.save_path)
                    print("Model was saved successfully")
            except:
                print("Failed save model")
            prev_loss = val_loss

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

    def clear_training_history(self):
        self.history = []

    def act(self, obs):
        obs = self.featurize(obs)
        board = np.expand_dims(obs['board'], axis=0)
        state = np.expand_dims(obs['state'], axis=0)
        res = self.sess.run(self.sampled_action, feed_dict={self.conv_ph: board, self.state_ph: state})
        return res

    def record_reward(self, reward):
        self.rewards.append(np.mean(reward))

    def close(self):
        self.sess.close()


# Simple wrapper around policy function to have an act function
class Expert:
    def __init__(self, config):
        self.__agent = SimpleAgent(config)

    def act(self, obs):
        return self.__agent.act(obs, None)

    def record_reward(self, reward):
        pass


class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


# Environment wrapper
class Stimulator:
    def __init__(self, env, config):
        self.env = env
        self.init(config)

    def init(self, config):
        self.env.seed(0)
        # Add 3 random agents
        agents = []
        for agent_id in range(3):
            agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

        # Add TensorforceAgent
        agent_id += 1
        agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
        self.env.set_agents(agents)
        self.env.set_training_agent(agents[-1].agent_id)
        self.env.set_init_game_state(None)

    def stimulate(self, agent, num_rollouts, render):
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('Iteration', i)
            obs = self.env.reset()[self.env.training_agent]
            done = False
            totalr = 0.
            steps = 0
            while not done:
                if render:
                    self.env.render()

                action = agent.act(obs)
                observations.append(obs)
                actions.append(action)

                obs = self.env.get_observations()
                all_actions = self.env.act(obs)
                all_actions.insert(self.env.training_agent, action)
                state, reward, done, _ = self.env.step(all_actions)

                obs = state[self.env.training_agent]
                r = reward[self.env.training_agent]
                totalr += r
                steps += 1

            print('rollout %i/%i return=%f' % (i + 1, num_rollouts, totalr))
            returns.append(totalr)
        print('Return summary: mean=%f, std=%f' % (np.mean(returns), np.std(returns)))
        agent.record_reward(returns)
        return (np.array(observations), np.array(actions))

    def label_obs(self, expert, obs):
        actions = []
        for o in obs:
            actions.append(expert.act(o))
        return np.array(actions)


def main():
    # Instantiate the environment
    config = ffa_v0_env()
    env = Pomme(**config["env_kwargs"])
    states = {
        "board": dict(shape=(BOARD_SIZE, BOARD_SIZE, 3,), type='float'),
        "state": dict(shape=(3,), type='float')
    }
    agent_dagger = Agent(env.action_space.n)
    # Load Expert
    expert = Expert(config["agent"](0, config["game_type"]))

    # Generate training data
    stimulator = Stimulator(env, config)
    training_data = stimulator.stimulate(expert, num_rollouts=num_rollouts, render=False)
    # Train DAgger Agent
    obs = training_data[0]
    labls = training_data[1]
    for i in range(2, 15):
        print("Train with DAgger, iter %i" % i)
        (stimulated_env, _) = stimulator.stimulate(agent_dagger, num_rollouts=num_rollouts, render=False)
        labels = stimulator.label_obs(expert, stimulated_env)
        obs = np.append(obs, stimulated_env, axis=0)
        labls = np.append(labls, labels, axis=0)
        agent_dagger.train(obs, labls, batch_size=batch_size, epochs=epochs)
    agent_dagger.close()
    env.close()


if __name__ == '__main__':
    main()

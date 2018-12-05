'''
    The policy for both self and opponent
    Based on PPO algorithm (OpenAI)

'''
#
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import copy
import numpy as np
import tensorflow as tf
from BasePolicy import policy
from collections import deque

MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3

ACTIONS = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]

class Policy_net:

    def  __init__(self, name: str, sess, ob_space, act_space, k=1, activation=tf.nn.relu, units=128):
        '''
        Network of PPO algorithm
        :param k: history used
        :param name: string
        :param sess:
        :param ob_space:
        :param act_space:
        :param activation:
        :param units:
        '''
        self.sess = sess

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, k * ob_space], name='obs')
            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=units, activation=activation)
                layer_2 = tf.layers.dense(inputs=layer_1, units=units, activation=activation)
                self.act_probs = tf.layers.dense(inputs=layer_2, units=act_space, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=units, activation=activation)
                layer_2 = tf.layers.dense(inputs=layer_1, units=units, activation=activation)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True, verbose=False):
        obs = np.array(obs)
        obs = obs[np.newaxis, :]
        if stochastic:
            act, v_preds = self.sess.run([self.act_stochastic, self.v_preds],
                                                    feed_dict={self.obs: obs})
            if verbose:
                pass
                # print('act_probs:', act_probs)
                print('act:{0}'.format(act))
                print('v_preds:', v_preds)
            return act[0], v_preds
        else:
            return self.sess.run([self.act_deterministic, self.v_preds],
                                 feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOTrain:

    def __init__(self, name, sess, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=0.9, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value: to clip ratio
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.sess = sess
        self.gamma = gamma
        self.lamda = 1.
        self.batch_size = 64
        self.epoch_num = 20
        self.clip_value = clip_value
        self.c_1 = c_1
        self.c_2 = c_2
        self.adam_lr = 2e-5
        self.adam_epsilon = 1e-5

        with tf.name_scope(name):
            pi_trainable = self.Policy.get_trainable_variables()
            old_pi_trainable = self.Old_Policy.get_trainable_variables()

            # assign_operations for policy parameter values to old policy parameters
            with tf.variable_scope('assign_op'):
                self.assign_ops = []
                for v_old, v in zip(old_pi_trainable, pi_trainable):
                    self.assign_ops.append(tf.assign(v_old, v))

            # inputs for train_op
            with tf.variable_scope('train_inp'):
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
                self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
                self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
                self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

            act_probs = self.Policy.act_probs
            act_probs_old = self.Old_Policy.act_probs

            # probabilities of actions which agent took with policy
            act = tf.one_hot(indices=self.actions, depth=len(ACTIONS))
            act_probs = act_probs * act
            act_probs = tf.reduce_sum(act_probs, axis=1)

            # probabilities of actions which agent took with old policy
            act_probs_old = act_probs_old * act
            act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

            with tf.variable_scope('loss'):
                # construct computation graph for loss_clip
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-8, 1.0))
                                - tf.log(tf.clip_by_value(act_probs_old, 1e-8, 1.0)))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                                  clip_value_max=1 + self.clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                loss_clip = -tf.reduce_mean(loss_clip)
                self.sum_clip = tf.summary.scalar('loss_clip', loss_clip)

                # construct computation graph for loss of entropy bonus
                entropy = -tf.reduce_sum(self.Policy.act_probs *
                                         tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
                entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
                self.sum_entropy = tf.summary.scalar('entropy', entropy)

                # construct computation graph for loss of value function
                v_preds = self.Policy.v_preds
                loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
                loss_vf = tf.reduce_mean(loss_vf)
                self.sum_vf = tf.summary.scalar('value_difference', loss_vf)

                # construct computation graph for loss
                self.total_loss = loss_clip + self.c_1 * loss_vf - self.c_2 * entropy
                self.sum_loss = tf.summary.scalar('total_loss', self.total_loss)

                self.g = tf.reduce_sum(self.rewards)
                self.sum_g = tf.summary.scalar('return', self.g)

            self.merged = tf.summary.merge([self.sum_clip, self.sum_vf, self.sum_loss, self.sum_g, self.sum_entropy])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, epsilon=self.adam_epsilon)

            self.gradients = optimizer.compute_gradients(self.total_loss, var_list=pi_trainable)

            self.train_op = optimizer.minimize(self.total_loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        _, total_loss = self.sess.run([self.train_op, self.total_loss], feed_dict={self.Policy.obs: obs,
                                                                                   self.Old_Policy.obs: obs,
                                                                                   self.actions: actions,
                                                                                   self.rewards: rewards,
                                                                                   self.v_preds_next: v_preds_next,
                                                                                   self.gaes: gaes})
        return total_loss

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next):
        return self.sess.run(self.merged, feed_dict={self.Policy.obs: obs,
                                                     self.Old_Policy.obs: obs,
                                                     self.actions: actions,
                                                     self.rewards: rewards,
                                                     self.v_preds_next: v_preds_next,
                                                     self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return self.sess.run(self.assign_ops)


    # discount reward
    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * self.lamda * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return self.sess.run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                        self.Old_Policy.obs: obs,
                                                        self.actions: actions,
                                                        self.rewards: rewards,
                                                        self.v_preds_next: v_preds_next,
                                                        self.gaes: gaes})

    def ppo_train(self, observations, actions, rewards, v_preds, v_preds_next, verbose=False):
        if verbose:
            print('PPO train now..........')
        gaes = self.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

        # convert list to numpy array for feeding tf.placeholder
        observations = np.array(observations).astype(dtype=np.float32)
        actions = np.array(actions).astype(dtype=np.int32)
        gaes = np.array(gaes).astype(dtype=np.float32)
        gaes = (gaes - gaes.mean()) / gaes.std()
        gaes = np.squeeze(gaes)
        rewards = np.array(rewards).astype(dtype=np.float32)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        v_preds_next = np.squeeze(v_preds_next)
        inp = [observations, actions, gaes, rewards, v_preds_next]

        self.assign_policy_parameters()
        # train
        for epoch in range(self.epoch_num):
            # sample indices from [low, high)
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=self.batch_size)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            total_loss = self.train(obs=sampled_inp[0],
                                    actions=sampled_inp[1],
                                    gaes=sampled_inp[2],
                                    rewards=sampled_inp[3],
                                    v_preds_next=sampled_inp[4])
            if verbose:
                print('total_loss:', total_loss)

        summary = self.get_summary(obs=inp[0],
                                   actions=inp[1],
                                   gaes=inp[2],
                                   rewards=inp[3],
                                   v_preds_next=inp[4])
        if verbose:
            print('PPO train end..........')
        return summary

class PPOPolicy(policy):

    '''
      agent : Agent name, different from model name
              We typically use agent name to infer this agent was trained with whom,
              while model name to infer the degree of our training
    '''
    def __init__(self,
                 k=1,
                 state_dim=8*4,
                 log_path='model/logs/',
                 model_path='model/latest.cpkt',
                 is_continuing=False,
                 is_training=True):

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.history_obs = deque(maxlen=k)

        self.k = k
        self.state_dim = state_dim
        self.empty_memory()

        with self.graph.as_default():

            self.pi = Policy_net('policy', self.sess, state_dim, len(ACTIONS), k)
            self.old_pi = Policy_net('old_policy', self.sess, state_dim, len(ACTIONS), k)

            self.PPOTrain = PPOTrain('train', self.sess, self.pi, self.old_pi)

            self.n_training = 0

        with self.sess.as_default():
            with self.graph.as_default():
                if is_training or is_continuing:
                    self.summary = tf.summary.FileWriter(log_path, self.sess.graph)

                # just using model
                if is_continuing or not is_training:
                    self.saver = tf.train.Saver()
                    self.load_model(model_path)

                # a totally new model
                else:
                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver()

    def choose_action(self, state):
        self.history_obs.append(state)
        k_obs = np.array([])
        for i in range(self.k):
            k_obs = np.hstack((k_obs, self.history_obs[i]))

        with self.sess.as_default():
            with self.graph.as_default():
                action, value = self.pi.act(k_obs)

        return action

    def get_action_value(self, state):
        self.history_obs.append(state)
        k_obs = np.array([])
        for i in range(self.k):
            k_obs = np.hstack((k_obs, self.history_obs[i]))

        with self.sess.as_default():
            with self.graph.as_default():
                action, value = self.pi.act(k_obs)

        return action, value

    def save_transition(self, obs, action, reward, v):

        self.history_obs.append(obs)
        k_obs = np.array([])
        for i in range(self.k):
            k_obs = np.hstack((k_obs, self.history_obs[i]))

        self.obs_memory.append(k_obs)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.v_memory.append(v)

    def empty_memory(self):

        self.history_obs.clear()
        for i in range(self.k):
            zero_state = np.zeros([self.state_dim])
            self.history_obs.append(zero_state)

        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.v_memory = []

    def train(self, final_obs):
        with self.sess.as_default():
            with self.graph.as_default():

                # compute the v of last obs
                # this obs was missed when done==True
                self.history_obs.append(final_obs)
                k_obs = np.array([])
                for i in range(self.k):
                    k_obs = np.hstack((k_obs, self.history_obs[i]))

                act, v = self.get_action_value(k_obs)

                # get v_next list
                v_next = self.v_memory[1:] + [v]

                summary = self.PPOTrain.ppo_train(self.obs_memory, self.action_memory, self.reward_memory, self.v_memory, v_next)

                self.n_training += 1
                self.summary.add_summary(summary, self.n_training)

                self.empty_memory()

    def save_model(self, path='model/latest.cpkt'):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, save_path=path)

    def load_model(self, path='model/latest.cpkt'):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, save_path=path)

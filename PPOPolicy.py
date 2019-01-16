'''
    The policy for both self and opponent
    Based on PPO algorithm (OpenAI)


'''

import copy
import numpy as np
import tensorflow as tf
from BasePolicy import policy
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3

ACTIONS = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]

class Policy_net:

    def  __init__(self, name: str, sess, ob_space, act_space, k=1, activation=tf.nn.relu, units=128, trainable=True):
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
                layer_1 = tf.layers.dense(inputs=self.obs, units=units * 2, activation=activation, trainable=trainable)
                layer_2 = tf.layers.dense(inputs=layer_1, units=units * 2, activation=activation, trainable=trainable)
                layer_3 = tf.layers.dense(inputs=layer_2, units=units, activation=activation, trainable=trainable)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space, activation=tf.nn.softmax, trainable=trainable)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=units * 2, activation=activation, trainable=trainable)
                layer_2 = tf.layers.dense(inputs=layer_1, units=units * 2, activation=activation, trainable=trainable)
                layer_3 = tf.layers.dense(inputs=layer_2, units=units, activation=activation, trainable=trainable)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None, trainable=trainable)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.scope = tf.get_variable_scope().name

    def act(self, obs):
        obs = np.array(obs)

        if obs.shape[0] != 1:
            obs = obs[np.newaxis, :]

        act_prob, act, v_preds = self.sess.run([self.act_probs, self.act_stochastic, self.v_preds],
                                                feed_dict={self.obs: obs})

        return act[0], act_prob[0][act[0]], v_preds[0, 0]

    def get_action_prob(self, obs):

        act_prob = self.sess.run(self.act_probs, feed_dict={self.obs: obs})

        return act_prob[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOTrain:

    def __init__(self, name, sess, Policy, Old_Policy, gamma=0.99, clip_value=0.2, c_1=0.5, c_2=0.05):
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
        self.lamda = 0.95
        self.batch_size = 256
        self.epoch_num = 10
        self.clip_value = clip_value
        self.c_1 = c_1
        self.c_2 = c_2
        self.adam_lr = 1e-4
        self.adam_epsilon = 1e-6

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
                ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                                - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                                  clip_value_max=1 + self.clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                loss_clip = -tf.reduce_mean(loss_clip)
                self.sum_clip = tf.summary.scalar('loss_clip', loss_clip)

                # self.sum_policy_w = tf.summary.histogram('policy_weight',
                #                                          self.sess.graph.get_tensor_by_name(
                #                                              'policy/policy_net/dense/kernel:0'))
                #
                # self.sum_value_w = tf.summary.histogram('value_weight',
                #                                         self.sess.graph.get_tensor_by_name(
                #                                             'policy/value_net/dense/kernel:0'))

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

            self.merged = tf.summary.merge_all()#([self.sum_clip, self.sum_vf, self.sum_loss, self.sum_g, self.sum_entropy, self.sum_value_w, self.sum_policy_w])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, epsilon=self.adam_epsilon)

            self.gradients = optimizer.compute_gradients(self.total_loss, var_list=pi_trainable)

            self.train_op = optimizer.minimize(self.total_loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        summary, _, total_loss = self.sess.run([self.merged, self.train_op, self.total_loss], feed_dict={self.Policy.obs: obs,
                                                                                   self.Old_Policy.obs: obs,
                                                                                   self.actions: actions,
                                                                                   self.rewards: rewards,
                                                                                   self.v_preds_next: v_preds_next,
                                                                                   self.gaes: gaes})
        return summary

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

    def ppo_train(self, observations, actions, rewards, gaes, v_preds_next, verbose=False):
        if verbose:
            print('PPO train now..........')

        self.assign_policy_parameters()

        actions = actions[:,np.newaxis]
        rewards = rewards[:,np.newaxis]
        gaes = gaes[:,np.newaxis]
        v_preds_next = v_preds_next[:,np.newaxis]

        #print(observations.shape)
        #print(actions.shape)
        #print(rewards.shape)
        #print(gaes.shape)
        #print(v_preds_next.shape)

        # concat and shuffle
        dataset = np.hstack((observations, actions, rewards, gaes, v_preds_next))
        np.random.shuffle(dataset)

        # recover
        observations = dataset[:, :164]
        actions = dataset[:, -4]
        rewards = dataset[:, -3]
        gaes = dataset[:, -2]
        v_preds_next = dataset[:, -1]

        actions = np.squeeze(actions)
        rewards = np.squeeze(rewards)
        gaes = np.squeeze(gaes)
        v_preds_next = np.squeeze(v_preds_next)

        l = len(actions)
        for i in range(self.epoch_num):
            start = 0
            end = start+self.batch_size
            while end < l:
                summary = self.train(observations[start:end],
                                     actions[start:end],
                                     gaes[start:end],
                                     rewards[start:end],
                                     v_preds_next[start:end])
                yield summary
                start += self.batch_size
                end += self.batch_size

        #print('an iteration ends')

        if verbose:
            print('PPO train end..........')

class PPOPolicy(policy):

    def __init__(self,
                 k=1,
                 state_dim=41,
                 log_path='model/logs/',
                 model_path='model/latest.cpkt',
                 is_continuing=False,
                 is_training=True):

        '''
        :param k: K-steps
        :param state_dim: here we use one-hot
        :param log_path:
        :param model_path:
        :param is_continuing: if continuing, we will create log and load model
        :param is_training: if training and not continuing, we won't load model
        '''

        self.model_path = model_path

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.2
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # trajectory data
        self.traj_obs      = []
        self.traj_act      = []
        self.traj_reward   = []
        self.traj_v        = []

        # training data
        self.obs_memory    = []
        self.act_memory    = []
        self.reward_memory = []
        self.v_next_memory = []
        self.gaes_memory   = np.array([]).astype(np.float32)

        self.k = k
        self.is_training = is_training
        self.state_dim = state_dim
        self.empty_all_memory()
        with self.sess.as_default():
            with self.graph.as_default():
                #with tf.device('/gpu:0'):
                self.pi = Policy_net('policy', self.sess, state_dim, len(ACTIONS), k, trainable=True)
                self.old_pi = Policy_net('old_policy', self.sess, state_dim, len(ACTIONS), k, trainable=False)

                self.PPOTrain = PPOTrain('train', self.sess, self.pi, self.old_pi)

                self.n_training = 0

        with self.sess.as_default():
            with self.graph.as_default():
                # with tf.device('/cpu:0'):
                if is_training or is_continuing:
                    self.summary = tf.summary.FileWriter(log_path, self.sess.graph)

                self.saver = tf.train.Saver(self.pi.get_trainable_variables())
                # just using model
                if is_continuing or not is_training:
                    self.sess.run(tf.global_variables_initializer())
                    self.load_model()

                # a totally new model
                else:
                    self.sess.run(tf.global_variables_initializer())


    def get_action(self, state):

        with self.sess.as_default():
            with self.graph.as_default():
                action,_, value = self.pi.act(state)

        return action

    def get_value(self, state):

        with self.sess.as_default():
            with self.graph.as_default():
                action,_, value = self.pi.act(state)

        return value

    def get_act_prob(self, state):
        if state.shape[0] != 1:
            state = state[np.newaxis, :]

        with self.sess.as_default():
            with self.graph.as_default():
                act_prob = self.pi.get_action_prob(state)

        return act_prob

    def get_action_value(self, state):

        with self.sess.as_default():
            with self.graph.as_default():
                action, act_prob, value = self.pi.act(state)

        return action, act_prob, value

    # def get_act_by_prob(self, prob):
    #     with self.sess.as_default():
    #         with self.graph.as_default():
    #             act = self.sess.run(self.act_select, feed_dict={self.act_prob: prob[np.newaxis, :]})
    #
    #     return act[0]

    # a transition should be saved by calling this function
    def save_transition(self, obs, next_obs, action, reward, v, done, t):

        # just save this transition
        if not done:

            self.traj_obs.append(obs)
            self.traj_act.append(action)
            self.traj_reward.append(reward)
            self.traj_v.append(v)

        # if done: add this trajectory to memory
        else:

            if t >= 2:

                self.traj_obs.append(obs)
                self.traj_act.append(action)
                self.traj_reward.append(reward)
                self.traj_v.append(v)

                self.obs_memory = self.obs_memory+self.traj_obs
                self.act_memory = self.act_memory + self.traj_act
                self.reward_memory = self.reward_memory + self.traj_reward

                # compute the v of last obs
                # this obs was missed when done==True
                act,_, v = self.get_action_value(np.hstack((obs[self.state_dim:], next_obs)))
                traj_v_next = self.traj_v[1:] + [v]
                traj_reward = self.traj_reward.copy()
                traj_v = self.traj_v.copy()

                gaes = self.PPOTrain.get_gaes(traj_reward, traj_v, traj_v_next)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean())/gaes.std()
                gaes = np.squeeze(gaes)

                self.gaes_memory = np.hstack((self.gaes_memory, gaes))
                self.v_next_memory = self.v_next_memory + traj_v_next
            else:
                pass

            self.empty_traj_memory()

    def empty_traj_memory(self):

        self.traj_obs.clear()
        self.traj_act.clear()
        self.traj_reward.clear()
        self.traj_v.clear()

    def empty_all_memory(self):

        self.empty_traj_memory()

        self.obs_memory.clear()
        self.act_memory.clear()
        self.reward_memory.clear()
        self.gaes_memory = np.array([]).astype(np.float32)
        self.v_next_memory.clear()

    def train(self, observations, actions, gaes, rewards, v_preds_next):

        with self.sess.as_default():
            with self.graph.as_default():

                # convert list to numpy array
                observation = np.vstack(observations).astype(np.float32)
                action = np.hstack(actions).astype(dtype=np.int32)
                gae = np.hstack(gaes).astype(np.float32)
                reward = np.hstack(rewards).astype(dtype=np.float32)
                v_pred_next = np.hstack(v_preds_next).astype(dtype=np.float32)

                for s in self.PPOTrain.ppo_train(observation, action, reward, gae, v_pred_next):
                    self.summary.add_summary(s, self.n_training)
                    self.n_training += 1

                #self.empty_all_memory()

    def save_model(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, save_path=self.model_path)

    def load_model(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, save_path=self.model_path)
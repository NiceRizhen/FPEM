import copy
import numpy as np
import tensorflow as tf
from BasePolicy import policy
from collections import deque
from PPOPolicy import PPOPolicy
from GameEnv import Game
import time

class Policy_net:

    def __init__(self, name: str, sess, state_dim, max_policy, activation=tf.nn.relu, trainable=True):
        '''
        Network of PPO algorithm
        :param k: history used
        :param name: string
        :param sess:
        :param ob_space:
        :param policy number:
        :param activation:
        :param units: neural network parameters
        '''
        self.sess = sess
        self.max_policy = max_policy

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='obs')
            self.policy_n = tf.placeholder(dtype=tf.int32, shape=[], name='cur_policy_n')
            self.policy_selected = tf.placeholder(dtype=tf.int32, shape=[None], name='policy_selected')

            with tf.variable_scope('weight'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=256, activation=activation, trainable=trainable)
                self.weight = tf.layers.dense(inputs=layer_1, units=max_policy, activation=None, trainable=trainable)

                self.policy_prob = tf.nn.softmax(self.weight[:, :self.policy_n+1], name='policy_prob')
                self.policy_argmax = tf.argmax(self.policy_prob, axis=1, name='policy_selected')

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=256, activation=activation, trainable=trainable)
                self.v_preds = tf.layers.dense(inputs=layer_1, units=1, activation=None, trainable=trainable)

            self.scope = tf.get_variable_scope().name

    def get_act_v(self, obs, cur_policy_n):
        obs = np.array(obs)

        if obs.shape[0] != 1:
            obs = obs[np.newaxis, :]

        policy, v = self.sess.run([self.policy_argmax, self.v_preds], feed_dict={self.obs: obs,
                                                                                 self.policy_n: cur_policy_n})

        return policy[0], v[0, 0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class PPOTrain:

    def __init__(self, name, sess, Policy, Old_Policy, max_policy, gamma=0.9, clip_value=0.2, c_1=1., c_2=0.01):
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
        self.max_policy = max_policy
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
                self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
                self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
                self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
                self.policy_selected = tf.placeholder(tf.int32, [None, ], name='policy_selected')

            with tf.variable_scope('loss'):

                with tf.variable_scope('weight_loss'):
                    policy_prob = tf.reduce_sum(tf.multiply(self.Policy.policy_prob, tf.one_hot(self.policy_selected, self.Policy.policy_n+1)), axis=1, name='policy_prob')
                    policy_prob_old = tf.reduce_sum(tf.multiply(self.Old_Policy.policy_prob, tf.one_hot(self.policy_selected, self.Old_Policy.policy_n+1)), axis=1, name='policy_prob_old')

                    # construct computation graph for weight loss
                    ratios = tf.exp(tf.log(tf.clip_by_value(policy_prob, 1e-10, 1)) -
                                    tf.log(tf.clip_by_value(policy_prob_old, 1e-10, 1)), name='ratios')
                    clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value,
                                                      clip_value_max=1 + clip_value, name='clip_ratios')
                    loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                    loss_clip = -tf.reduce_mean(loss_clip)
                    tf.summary.scalar('weight_loss', loss_clip)

                # to show whether params are updated
                tf.summary.histogram('policy_weight',
                    self.sess.graph.get_tensor_by_name(
                            'mix_policy/weight/dense/kernel:0'))

                tf.summary.histogram('value_weight',
                                        self.sess.graph.get_tensor_by_name(
                                        'mix_policy/value_net/dense/kernel:0'))

                # construct computation graph for loss of entropy bonus
                with tf.variable_scope('entropy_loss'):
                    entropy = -tf.reduce_sum(self.Policy.policy_prob *
                                             tf.log(tf.clip_by_value(self.Policy.policy_prob, 1e-10, 1.0)), axis=1)
                    entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
                    self.sum_entropy = tf.summary.scalar('entropy', entropy)

                with tf.variable_scope('value_loss'):
                    # construct computation graph for loss of value function
                    v_preds = self.Policy.v_preds
                    loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
                    loss_vf = tf.reduce_mean(loss_vf)
                    self.sum_vf = tf.summary.scalar('value_difference', loss_vf)

                with tf.variable_scope('total_loss'):
                    # construct computation graph for loss
                    self.total_loss = loss_clip + self.c_1 * loss_vf - self.c_2 * entropy
                    self.sum_loss = tf.summary.scalar('total_loss', self.total_loss)

                g = tf.reduce_sum(self.rewards)
                tf.summary.scalar('return', g)

            self.merged = tf.summary.merge_all()  # ([self.sum_clip, self.sum_vf, self.sum_loss, self.sum_g, self.sum_policy_w, self.sum_value_w, self.sum_ratios])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, epsilon=self.adam_epsilon)

            self.gradients = optimizer.compute_gradients(self.total_loss, var_list=pi_trainable)

            self.train_op = optimizer.minimize(self.total_loss, var_list=pi_trainable)

    def train(self, obs, policy, gaes, rewards, v_preds_next, cur_policy_n):

        summary, _ = self.sess.run([self.merged, self.train_op],
                                    feed_dict={self.Policy.obs:obs,
                                               self.Old_Policy.obs:obs,
                                               self.rewards: rewards,
                                               self.policy_selected:policy,
                                               self.v_preds_next: v_preds_next,
                                               self.gaes: gaes,
                                               self.Policy.policy_n: cur_policy_n,
                                               self.Old_Policy.policy_n: cur_policy_n})
        return summary


    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return self.sess.run(self.assign_ops)

    # generalized advantage estimated
    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * self.lamda * gaes[t + 1]
        return gaes

    def get_grad(self, obs, policy, gaes, rewards, v_preds_next):
        return self.sess.run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                        self.Old_Policy.obs: obs,
                                                        self.policy_selected: policy,
                                                        self.rewards: rewards,
                                                        self.v_preds_next: v_preds_next,
                                                        self.gaes: gaes})

    def ppo_train(self, observations, policy, rewards, gaes, v_preds_next, cur_policy_n, verbose=False):
        if verbose:
            print('PPO train now..........')

        self.assign_policy_parameters()

        rewards = rewards[:, np.newaxis]
        policy = policy[:, np.newaxis]
        gaes = gaes[:, np.newaxis]
        v_preds_next = v_preds_next[:, np.newaxis]

        #print(observations.shape)
        #print(policy.shape)
        #print(gaes.shape)
        #print(rewards.shape)
        #print(v_preds_next.shape)

        dataset = np.hstack((observations, policy, rewards, gaes, v_preds_next))
        np.random.shuffle(dataset)

        observations = dataset[:, :164]
        policy = dataset[:, -4]
        rewards = dataset[:, -3]
        gaes = dataset[:, -2]
        v_preds_next = dataset[:, -1]

        rewards = np.squeeze(rewards)
        policy = np.squeeze(policy)
        gaes = np.squeeze(gaes)
        v_preds_next = np.squeeze(v_preds_next)

        l = len(rewards)
        for i in range(self.epoch_num):
            start = 0
            end = start + self.batch_size

            while end < l:
                summary = self.train(observations[start:end],
                                     policy[start:end],
                                     gaes[start:end],
                                     rewards[start:end],
                                     v_preds_next[start:end],
                                     cur_policy_n = cur_policy_n)

                #yield summary
                start += self.batch_size
                end += self.batch_size

        #print('an iteration ends')

        if verbose:
            print('PPO train end..........')


# a mla to mix base policy by action
class MixPolicy(policy):

    def __init__(self,
                 max_policy,
                 k=1,
                 state_dim=41,
                 log_path='model/logs/',
                 model_path='model/policy/latest.ckpt',
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
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.model_path = model_path

        # trajectory data
        self.traj_obs = []
        self.traj_policy = []
        self.traj_reward = []
        self.traj_v = []

        # training data
        self.obs_memory = []
        self.policy_memory = []
        self.reward_memory = []
        self.v_next_memory = []
        self.gaes_memory = np.array([]).astype(np.float32)

        self.k = k
        self.is_training = is_training
        self.empty_all_memory()
        self.state_dim = k * state_dim

        with self.graph.as_default():

            self.pi = Policy_net('mix_policy', self.sess, k * state_dim, max_policy, trainable=True)
            self.old_pi = Policy_net('old_mix_policy', self.sess, k * state_dim, max_policy, trainable=False)

            self.PPOTrain = PPOTrain('mix_policy_train', self.sess, self.pi, self.old_pi, max_policy=max_policy)

            self.n_training = 0

        with self.sess.as_default():
            with self.graph.as_default():
                # with tf.device('/cpu:0'):
                if is_training or is_continuing:
                    self.summary = tf.summary.FileWriter(log_path, self.sess.graph)

                # just using model
                if is_continuing or not is_training:
                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver(self.pi.get_trainable_variables())
                    self.load_model()

                # a totally new model
                else:
                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver(self.pi.get_trainable_variables())

    def get_policy(self, state, cur_policy_n):
        with self.sess.as_default():
            with self.graph.as_default():
                policy, v = self.pi.get_act_v(state,cur_policy_n)

        return policy, v

    def save_transition(self, obs, next_obs, policy, reward, v, done, t, cur_policy_n):
        """
        A transition should be saved by calling this function
        We call it just when we are training a model.

        :param obs: state or observation
        :param next_obs:
        :param p_select: the policy choosen to get action
        :param reward:
        :param v: value calculated by value function
        :param done: whether this epoch ends
        :param t: the time-step number of this epoch
        :return: None with transition been saved
        """

        # just save this transition
        if not done:
            self.traj_obs.append(obs)
            self.traj_policy.append(policy)
            self.traj_reward.append(reward)
            self.traj_v.append(v)

        # if done: add this trajectory to memory
        else:
            if t >= 2:

                self.traj_obs.append(obs)
                self.traj_policy.append(policy)
                self.traj_reward.append(reward)
                self.traj_v.append(v)

                self.obs_memory = self.obs_memory + self.traj_obs
                self.policy_memory = self.policy_memory + self.traj_policy
                self.reward_memory = self.reward_memory + self.traj_reward

                # compute the v of last obs
                # this obs was missed when done==True
                policy, v = self.get_policy(np.hstack((obs[int(self.state_dim/self.k) :], next_obs)), cur_policy_n)
                traj_v_next = self.traj_v[1:] + [v]
                traj_reward = self.traj_reward.copy()
                traj_v = self.traj_v.copy()

                gaes = self.PPOTrain.get_gaes(traj_reward, traj_v, traj_v_next)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()
                gaes = np.squeeze(gaes)

                self.gaes_memory = np.hstack((self.gaes_memory, gaes))
                self.v_next_memory = self.v_next_memory + traj_v_next
            else:
                pass

            self.empty_traj_memory()

    def empty_traj_memory(self):

        self.traj_obs = []
        self.traj_policy = []
        self.traj_reward = []
        self.traj_v = []

    def empty_all_memory(self):

        self.empty_traj_memory()

        self.obs_memory = []
        self.policy_memory = []
        self.reward_memory = []
        self.gaes_memory = np.array([]).astype(np.float32)
        self.v_next_memory = []

    def train(self, observations, policys, gaes, rewards, v_preds_next, cur_policy_n):

        with self.sess.as_default():
            with self.graph.as_default():

                # convert list to numpy array
                observation = np.vstack(observations).astype(np.float32)
                policy = np.hstack(policys).astype(dtype=np.int32)
                gae = np.hstack(gaes).astype(np.float32)
                reward = np.hstack(rewards).astype(dtype=np.float32)
                v_pred_next = np.hstack(v_preds_next).astype(dtype=np.float32)

                self.PPOTrain.ppo_train(observation, policy, reward, gae, v_pred_next, cur_policy_n)
                    # self.summary.add_summary(s, self.n_training)
                    # self.n_training += 1

                self.empty_all_memory()

    def save_model(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, save_path=self.model_path)

    def load_model(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, save_path=self.model_path)

'''
  We try to mix policy with an reinforcement learning method.
  This file provides a policy gradient method.
  In our implementation, we don't just use act prob to
  maximize expected return, but replace it with ∑wπ.

  MixPolicyBA: mix policy by action
  MixPolicyBP: mix policy by policy

  @Author: Jingcheng Pang
'''

import numpy as np
import tensorflow as tf
from PPOPolicy import PPOPolicy
from GameEnv import Game
from collections import deque
from MixPolicyPPO import MixPolicy

class MixPolicyBA:
    def __init__(
            self,
            policy_n,
            input_dim=4*45,
            epoch_num=10,
            batch_size=128,
            learning_rate=1e-4,
            reward_decay=0.98,
            log_path='model/mix/logs/',
            model_path='model/mix/latest.ckpt',
            is_continuing=False,
            is_training=True
    ):
        '''
        It's a policy gradient algorithm's variety that maximize expected return

        :param policy_n: number of policy in policy set
        :param input_dim: observation's dimension
        :param epoch_num: times of one training
        :param batch_size: a batch
        :param learning_rate:
        :param reward_decay:
        :param log_path:
        :param model_path:
        :param is_continuing: whether you are going on
        :param is_training: if not training, we will load model
        '''
        self.input_dim = input_dim
        self.policy_n = policy_n
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epoch_num = epoch_num
        self.batch_size = batch_size

        # represent for obs act_pro reward respectively
        self.obs_memory, self.ap_memory, self.rs_memory = [], [], []
        self.all_obs, self.all_ap, self.all_rs, self.dis_rs = [], [], [], []

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.n_training = 0

        with self.sess.as_default():
            with self.graph.as_default():

                self._build_net()

                if is_training or is_continuing:
                    self.summary = tf.summary.FileWriter(log_path, self.sess.graph)

                # just using exist model
                if is_continuing or not is_training:
                    self.saver = tf.train.Saver(self.get_trainable_variables())
                    self.load_model(model_path)

                # a totally new model
                else:
                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver(self.get_trainable_variables())

    def _build_net(self):
        with tf.variable_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.input_dim], name='observations')
            self.act_prob = tf.placeholder(tf.float32, [None, self.policy_n], name='act_prob') # action prob calculated by every pi
            self.dis_reward = tf.placeholder(tf.float32, [None, ], name='dis_reward')
            self.reward = tf.placeholder(tf.float32, [None,], name='reward')

        with tf.variable_scope('policy_net'):

            layer = tf.layers.dense(
                inputs=self.obs,
                units=256,
                activation=tf.nn.relu,
                name='dense1'
            )

            self.weight = tf.layers.dense(
                inputs=layer,
                units=self.policy_n,
                activation=tf.nn.softmax,
                name='output'
            )

        self.scope = tf.get_variable_scope().name

        with tf.variable_scope('loss'):
            p = tf.reduce_sum(tf.multiply(self.weight, self.act_prob), axis=1)
            loss = tf.reduce_mean((-tf.log(tf.clip_by_value(p,1e-10, 1))) * self.dis_reward)

        with tf.variable_scope('summary'):
            sum_v = tf.summary.scalar('return', tf.reduce_sum(self.reward))
            sum_loss = tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge([sum_v, sum_loss])

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_act_prob_full_state(self, observation):

        with self.sess.as_default():
            with self.graph.as_default():
                weight = self.sess.run(self.weight, feed_dict={self.obs: observation[np.newaxis, :]})

        return weight

    def save_transition(self, obs, act_prob, reward, done):

        if not done:

            self.obs_memory.append(obs)
            self.ap_memory.append(act_prob)
            self.rs_memory.append(reward)

        else:

            self.obs_memory.append(obs)
            self.ap_memory.append(act_prob)
            self.rs_memory.append(reward)

            self.dis_rs = self.dis_rs + self._discount_and_norm_rewards().tolist()
            self.all_obs = self.all_obs + self.obs_memory
            self.all_ap = self.all_ap + self.ap_memory
            self.all_rs = self.all_rs + self.rs_memory

            self.empty_traj_memory()

    def empty_traj_memory(self):

        self.obs_memory.clear()
        self.ap_memory.clear()
        self.rs_memory.clear()

    def empty_all_memory(self):
        self.obs_memory, self.ap_memory, self.rs_memory = [], [], []
        self.all_obs, self.all_ap, self.all_rs, self.dis_rs = [], [], [], []

    def train(self):

        observations = np.array(self.all_obs).astype(dtype=np.float32)
        act_prob = np.array(self.all_ap)
        rs = np.array(self.all_rs)
        dis_r = np.array(self.dis_rs).astype(dtype=np.float32)

        rs = rs[:, np.newaxis]
        dis_r = dis_r[:,np.newaxis]

        print(observations.shape)
        print(act_prob.shape)
        print(rs.shape)
        print(dis_r.shape)

        try:
            dataset = np.hstack((observations, act_prob, rs, dis_r))
            np.random.shuffle(dataset)

            observations = dataset[:, :180]
            act_prob = dataset[:, 180:185]
            rs = dataset[:, -2]
            dis_r = dataset[:, -1]

            rs = np.squeeze(rs)
            dis_r = np.squeeze(dis_r)

            l = len(rs)
            # train on batch
            for i in range(self.epoch_num):
                start = 0
                end = start+self.batch_size
                while end < l:
                    _, summary = self.sess.run([self.train_op, self.merged], feed_dict={
                        self.obs: observations[start:end],
                        self.act_prob: act_prob[start:end],
                        self.dis_reward: dis_r[start:end],
                        self.reward: rs[start:end]
                    })

                    self.summary.add_summary(summary, self.n_training)
                    self.n_training += 1

                    start += self.batch_size
                    end += self.batch_size

            self.empty_all_memory()

        except Exception as err:
            print(err)
        finally:
            self.empty_all_memory()
            return

    def _discount_and_norm_rewards(self):

        discounted_ep_rs = np.zeros_like(self.rs_memory, dtype=np.float32)

        add = 0
        # self.rs_memory.reverse()
        # for reward in self.rs_memory:

        for t in reversed(range(0, len(self.rs_memory))):
            add = add * self.gamma + self.rs_memory[t]
            discounted_ep_rs[t] = add

        return discounted_ep_rs

    def save_model(self, path='model/mix/latest.cpkt'):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, save_path=path)

    def load_model(self, path='model/mix/latest.cpkt'):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, save_path=path)

class MixPolicyBP:
    def __init__(
            self,
            policy_n,
            input_dim=4*45,
            epoch_num=10,
            batch_size=128,
            learning_rate=1e-4,
            reward_decay=0.98,
            log_path='model/mix/logs/',
            model_path='model/mix/latest.ckpt',
            is_continuing=False,
            is_training=True
    ):
        '''
        It's a policy gradient algorithm's variety that maximize expected return

        :param policy_n: number of policy in policy set
        :param input_dim: observation's dimension
        :param epoch_num: times of one training
        :param batch_size: a batch
        :param learning_rate:
        :param reward_decay:
        :param log_path:
        :param model_path:
        :param is_continuing: whether you are going on
        :param is_training: if not training, we will load model
        '''
        self.input_dim = input_dim
        self.policy_n = policy_n
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epoch_num = epoch_num
        self.batch_size = batch_size

        # represent for obs act_pro reward respectively
        self.obs_memory, self.ps_memory, self.rs_memory = [], [], []
        self.all_obs, self.all_ps, self.all_rs, self.dis_rs = [], [], [], []

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.n_training = 0

        with self.sess.as_default():
            with self.graph.as_default():

                self._build_net()

                if is_training or is_continuing:
                    self.summary = tf.summary.FileWriter(log_path, self.sess.graph)

                # just using exist model
                if is_continuing or not is_training:
                    self.saver = tf.train.Saver(self.get_trainable_variables())
                    self.load_model(model_path)

                # a totally new model
                else:
                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver(self.get_trainable_variables())

    def _build_net(self):
        with tf.variable_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.input_dim], name='observations')
            self.dis_reward = tf.placeholder(tf.float32, [None, ], name='dis_reward')
            self.p_select = tf.placeholder(tf.int32, [None, ], name='policy_selected')
            self.reward = tf.placeholder(tf.float32, [None,], name='reward')

        with tf.variable_scope('policy_net'):

            layer = tf.layers.dense(
                inputs=self.obs,
                units=256,
                activation=tf.nn.relu,
                name='dense1'
            )

            self.weight = tf.layers.dense(
                inputs=layer,
                units=self.policy_n,
                activation=tf.nn.softmax,
                name='output'
            )

        self.scope = tf.get_variable_scope().name

        with tf.variable_scope('loss'):
            logpi = tf.reduce_sum(-tf.log(self.weight) * tf.one_hot(self.p_select, self.policy_n), axis=1)
            loss = tf.reduce_mean(logpi * self.dis_reward)

        with tf.variable_scope('summary'):
            sum_v = tf.summary.scalar('return', tf.reduce_sum(self.reward))
            sum_loss = tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge([sum_v, sum_loss])

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_act_prob_full_state(self, observation):

        with self.sess.as_default():
            with self.graph.as_default():
                weight = self.sess.run(self.weight, feed_dict={self.obs: observation[np.newaxis, :]})

        return weight

    def save_transition(self, obs, p_selected, reward, done):

        if not done:

            self.obs_memory.append(obs)
            self.ps_memory.append(p_selected)
            self.rs_memory.append(reward)

        else:

            self.obs_memory.append(obs)
            self.ps_memory.append(p_selected)
            self.rs_memory.append(reward)

            self.dis_rs = self.dis_rs + self._discount_and_norm_rewards().tolist()
            self.all_obs = self.all_obs + self.obs_memory
            self.all_ps = self.all_ps + self.ps_memory
            self.all_rs = self.all_rs + self.rs_memory

            self.empty_traj_memory()

    def empty_traj_memory(self):

        self.obs_memory.clear()
        self.ps_memory.clear()
        self.rs_memory.clear()

    def empty_all_memory(self):
        self.obs_memory, self.ps_memory, self.rs_memory = [], [], []
        self.all_obs, self.all_ps, self.all_rs, self.dis_rs = [], [], [], []

    def train(self):

        observations = np.array(self.all_obs).astype(dtype=np.float32)
        ps = np.array(self.all_ps)
        rs = np.array(self.all_rs)
        dis_r = np.array(self.dis_rs).astype(dtype=np.float32)

        ps = ps[:, np.newaxis]
        rs = rs[:, np.newaxis]
        dis_r = dis_r[:,np.newaxis]

        print(observations.shape)
        print(ps.shape)
        print(rs.shape)
        print(dis_r.shape)

        try:
            dataset = np.hstack((observations, ps, rs, dis_r))
            np.random.shuffle(dataset)

            observations = dataset[:, :180]
            ps = dataset[:, -3]
            rs = dataset[:, -2]
            dis_r = dataset[:, -1]

            ps = np.squeeze(ps)
            rs = np.squeeze(rs)
            dis_r = np.squeeze(dis_r)

            l = len(rs)
            # train on batch
            for i in range(self.epoch_num):
                start = 0
                end = start+self.batch_size
                while end < l:
                    _, summary = self.sess.run([self.train_op, self.merged], feed_dict={
                        self.obs: observations[start:end],
                        self.p_select: ps[start:end],
                        self.dis_reward: dis_r[start:end],
                        self.reward: rs[start:end]
                    })

                    self.summary.add_summary(summary, self.n_training)
                    self.n_training += 1

                    start += self.batch_size
                    end += self.batch_size

            self.empty_all_memory()

        except Exception as err:
            print(err)
        finally:
            self.empty_all_memory()
            return

    def _discount_and_norm_rewards(self):

        discounted_ep_rs = np.zeros_like(self.rs_memory, dtype=np.float32)

        add = 0
        # self.rs_memory.reverse()
        # for reward in self.rs_memory:

        for t in reversed(range(0, len(self.rs_memory))):
            add = add * self.gamma + self.rs_memory[t]
            discounted_ep_rs[t] = add

        return discounted_ep_rs

    def save_model(self, path='model/mix/latest.cpkt'):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.save(self.sess, save_path=path)

    def load_model(self, path='model/mix/latest.cpkt'):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, save_path=path)

# mix policy by action
if __name__ == "__main__":

    env = Game(8,8)

    # get policy set
    ps1, ps2 = [], []

    for i in range(5):
        ps1.append(
            PPOPolicy(is_training=False, k=4, model_path='model/1-3(150000)/{0}.ckpt'.format(i + 1))
        )

        ps2.append(
            PPOPolicy(is_training=False, k=4, model_path='model/2-3(150000)/{0}.ckpt'.format(i + 1))
        )

    ps1_n = len(ps1)
    ps2_n = len(ps2)

    mix_policy1 = MixPolicy(ps1_n, log_path='model/mix/log1', k=4)
    mix_policy2 = MixPolicy(ps2_n, log_path='model/mix/log2', k=4)

    p1_state = deque(maxlen=4)
    p2_state = deque(maxlen=4)

    # set training params
    epoch = 1
    _train_num = 1000
    who_is_training = 1
    _save_num = 40000
    _save_times = 1

    while True:
        t = 0

        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        for i in range(4):
            zero_state = np.zeros([45])
            p1_state.append(zero_state)
            p2_state.append(zero_state)

        while True:

            p1_state.append(s1)
            p2_state.append(s2)

            state1 = np.array([])
            for obs in p1_state:
                state1 = np.hstack((state1, obs))

            state2 = np.array([])
            for obs in p2_state:
                state2 = np.hstack((state2, obs))

            w1, v1 = mix_policy1.get_weight(state1)
            w2, v2 = mix_policy2.get_weight(state2)

            player1_pi = []
            player2_pi = []

            # select action1
            for p in ps1:
                pi_prob = p.get_action_prob_full_state(state1)
                player1_pi.append(pi_prob)

            act_prob = np.array(player1_pi)
            act_prob = np.matmul(w1, act_prob).squeeze()

            act_prob /=act_prob.sum()
            a1 = ps1[4].get_act_by_prob(act_prob)

            # select action2
            for p in ps2:
                pi_prob = p.get_action_prob_full_state(state2)
                player2_pi.append(pi_prob)

            act_prob = np.array(player2_pi)
            act_prob = np.matmul(w2, act_prob).squeeze()

            act_prob /= act_prob.sum()
            a2 = ps1[4].get_act_by_prob(act_prob)

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            if who_is_training == 1:
                mix_policy1.save_transition(state1, s1_, [p[a1] for p in player1_pi], r1, v1, done, t)
            else:
                mix_policy2.save_transition(state2, s2_, [p[a2] for p in player2_pi], r2, v2, done, t)

            s1 = s1_
            s2 = s2_
            t += 1

            if t > 100:
                if who_is_training == 1:
                    mix_policy1.empty_traj_memory()
                else:
                    mix_policy2.empty_traj_memory()

                break

            if done:
                epoch += 1
                break

        p1_state.clear()
        p2_state.clear()

        if epoch % _train_num == 0:
            if who_is_training == 1:
                mix_policy1.train()
            else:
                mix_policy2.train()

            who_is_training = 1 if who_is_training == 2 else 2
            mix_policy1.empty_all_memory()
            mix_policy2.empty_all_memory()

            epoch += 1

        if epoch % _save_num == 0:
            mix_policy1.save_model('model/mix/p1/{0}.ckpt'.format(_save_times))
            mix_policy2.save_model('model/mix/p2/{0}.ckpt'.format(_save_times))
            _save_times += 1

# mix policy by policy
# if __name__ == "__main__":
#
#     env = Game(8,8)
#
#     # get policy set
#     ps1, ps2 = [], []
#
#     for i in range(5):
#         ps1.append(
#             PPOPolicy(is_training=False, k=4, model_path='model/1-3(150000)/{0}.ckpt'.format(i + 1))
#         )
#
#         ps2.append(
#             PPOPolicy(is_training=False, k=4, model_path='model/2-3(150000)/{0}.ckpt'.format(i + 1))
#         )
#
#     ps1_n = len(ps1)
#     ps2_n = len(ps2)
#
#     mix_policy1 = MixPolicyBP(ps1_n, log_path='model/mix/log1')
#     mix_policy2 = MixPolicyBP(ps2_n, log_path='model/mix/log2')
#
#     p1_state = deque(maxlen=4)
#     p2_state = deque(maxlen=4)
#
#     # set training params
#     epoch = 1
#     _train_num = 100
#     _change_num = 40000
#     who_is_training = 1
#     iteration = 1
#
#     while True:
#         t = 0
#
#         if epoch % _change_num == 0:
#             epoch += 1 # then player won't change continuously
#             who_is_training = 1 if who_is_training == 2 else 2
#             mix_policy1.empty_all_memory()
#             mix_policy2.empty_all_memory()
#
#             if who_is_training == 2:
#                 mix_policy1.save_model('model/mix/player2/{0}.ckpt'.format(iteration))
#
#             if who_is_training == 1:
#                 mix_policy2.save_model('model/mix/player1/{0}.ckpt'.format(iteration))
#                 iteration += 1
#
#         s, _ = env.reset()
#         s1 = s[0]
#         s2 = s[1]
#
#         for i in range(4):
#             zero_state = np.zeros([45])
#             p1_state.append(zero_state)
#             p2_state.append(zero_state)
#
#         while True:
#
#             p1_state.append(s1)
#             p2_state.append(s2)
#
#             state1 = np.array([])
#             for obs in p1_state:
#                 state1 = np.hstack((state1, obs))
#
#             state2 = np.array([])
#             for obs in p2_state:
#                 state2 = np.hstack((state2, obs))
#
#             w1 = mix_policy1.get_act_prob_full_state(state1)
#             w2 = mix_policy2.get_act_prob_full_state(state2)
#
#             player1_pi = []
#             player2_pi = []
#
#             # select action1
#             index1 = np.argmax(w1[0])
#             prob = ps1[index1].get_action_prob_full_state(state1)
#             a1 = ps1[4].get_act_by_prob(prob)
#
#             # select action2
#             index2 = np.argmax(w2[0])
#             prob = ps2[index2].get_action_prob_full_state(state2)
#             a2 = ps1[4].get_act_by_prob(prob)
#
#             s1_, s2_, r1, r2, done = env.step(a1, a2)
#
#             if who_is_training == 1:
#                 mix_policy1.save_transition(state1, index1, r1, done)
#             else:
#                 mix_policy2.save_transition(state2, index2, r2, done)
#
#             s1 = s1_
#             s2 = s2_
#             t += 1
#
#             if t > 100:
#                 if who_is_training == 1:
#                     mix_policy1.empty_traj_memory()
#                 else:
#                     mix_policy2.empty_traj_memory()
#
#                 break
#
#             if done:
#                 epoch += 1
#                 break
#
#         p1_state.clear()
#         p2_state.clear()
#
#         if epoch % _train_num == 0:
#             if who_is_training == 1:
#                 mix_policy1.train()
#             else:
#                 mix_policy2.train()
#
#             # prevent a player from training twice
#             if epoch % _change_num != 0:
#                 epoch += 1
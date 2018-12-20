'''
  We try to mix policy with racos method.

  @Author: Jingcheng Pang
'''

import numpy as np
from GameEnv import Game
from PPOPolicy import PPOPolicy
from multiprocessing import Process, Manager
from collections import deque
from zoopt import Dimension, Objective, Parameter, Opt

class evaluate_obj():

    def __init__(self, who_is_tested):

        # _w_waited is the weight waiting tested
        # while _w_with is the weight that _w_waited training with
        self._w1 = [0.2, 0.2, 0.2, 0.2, 0.2]
        self._w2 = [0.2, 0.2, 0.2, 0.2, 0.2]
        self._who_is_tested = who_is_tested

    def set_w(self, opponent_w):
        opponent_w = [a/sum(opponent_w) for a in opponent_w]
        if self._who_is_tested == 1:
            self._w2 = opponent_w
        else:
            self._w1 = opponent_w

    def evaluate_Mixpolicy(self, w1, w2, re):

        env = Game(8, 8)

        # init policy set
        ps1 = []
        ps2 = []

        for i in range(5):
            ps1.append(
                PPOPolicy(is_training=False, k=4, model_path='model/1-3(150000)/{0}.ckpt'.format(i + 1))
            )

            ps2.append(
                PPOPolicy(is_training=False, k=4, model_path='model/2-3(150000)/{0}.ckpt'.format(i + 1))
            )

        n_p1 = len(ps1)
        n_p2 = len(ps2)

        reward = 0

        epoch = 0
        while epoch < 1500:
            t = 0
            episode = []
            epoch += 1

            s, _ = env.reset()
            s1 = s[0]
            s2 = s[1]

            index1 = np.random.choice(n_p1, 1, p=w1)[0]
            index2 = np.random.choice(n_p2, 1, p=w2)[0]

            p1 = ps1[index1]
            p2 = ps2[index2]

            p1_state = deque(maxlen=4)
            p2_state = deque(maxlen=4)

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

                a1 = p1.choose_action_full_state(state1)
                a2 = p2.choose_action_full_state(state2)

                s1_, s2_, r1, r2, done = env.step(a1, a2)

                if self._who_is_tested == 1:
                    episode.append(r1)
                else:
                    episode.append(r2)

                s1 = s1_
                s2 = s2_
                t += 1

                if t > 200:
                    epoch -= 1
                    break

                if done:
                    break

            p1_state.clear()
            p2_state.clear()

            if done:
                reward += sum(episode)

        reward = reward/150
        re.append(reward)

        del env

        for p in ps1:
            p.sess.close()
        for p in ps2:
            p.sess.close()

    # evaluate with multi-process
    def evaluate(self, solution):

        x = solution.get_x()
        x = [a/sum(x) for a in x]
        manager = Manager()

        # processes number
        K = 4

        # init evaluate reward
        reward = manager.list([])

        if self._who_is_tested == 1:
            # multi-process start!
            process_list = []

            for i in range(K):
                p = Process(target=self.evaluate_Mixpolicy, args=(x, self._w2, reward,))
                process_list.append(p)

            for p in process_list:
                p.start()

            for p in process_list:
                p.join()

        else:
            # multi-process start!
            process_list = []

            for i in range(K):
                p = Process(target=self.evaluate_Mixpolicy, args=(self._w1, x, reward,))
                process_list.append(p)

            for p in process_list:
                p.start()

            for p in process_list:
                p.join()

        # print iteration information
        reward = sum(reward)/K

        return -reward


if __name__ == "__main__":
    dim = 5  # dimension

    _eva_class1 = evaluate_obj(1)
    _eva_class2 = evaluate_obj(2)

    _best_w1 = [1,1,1,1,1]
    _best_w2 = [1,1,1,1,1]
    _last_w1 = [0.1,0.1,0.1,0.1,0.1]
    _last_w2 = [0.1,0.1,0.1,0.1,0.1]

    iteration = 1
    while True:
        print('iteration:{0}'.format(iteration))

        # optimize player1's weight
        print('optimize weight1')
        _eva_class1.set_w(_best_w2)
        obj = Objective(_eva_class1.evaluate, Dimension(dim, [[0, 1]] * dim, [True] * dim))
        solution = Opt.min(obj, Parameter(budget=4 * dim))
        solution.print_solution()
        _last_w1 = _best_w1
        _best_w1 = solution.get_x()

        # optimize player2's weight
        print('optimize weight2')
        _eva_class2.set_w(_best_w1)
        obj = Objective(_eva_class2.evaluate, Dimension(dim, [[0, 1]] * dim, [True] * dim))
        solution = Opt.min(obj, Parameter(budget=4 * dim))
        solution.print_solution()
        _last_w2 = _best_w2
        _best_w2 = solution.get_x()

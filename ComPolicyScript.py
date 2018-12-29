'''
  This file is a script file for Comparing different policy.

  @Author: Jingcheng Pang
'''

import numpy as np
from GameEnv import Game
import matplotlib.pyplot as plt
from PPOPolicy import PPOPolicy
import random
#
# def run_with_w(p1, p2):
#
#     epoch = 0
#     reward = []
#
#     while epoch < 100:
#
#         t = 0
#         episode1 = []
#         episode2 = []
#         p = random.choice(p2)
#
#         s, _ = env.reset()
#         s1 = s[0]
#         s2 = s[1]
#
#         while True:
#
#             a1 = p1.choose_action(s1)
#             a2 = p2.choose_action(s2)
#             s1_, s2_, r1, r2, done = env.step(a1, a2)
#             episode1.append(r1)
#             episode2.append(r2)
#             s1 = s1_
#             s2 = s2_
#             t += 1
#             if t % 300 == 0:
#                 reward.append(sum(episode))
#                 episode = []
#                 epoch += 1
#                 t = 0
#
#     return reward



def run_with_policy(pi1, pi2):

    epoch = 0
    reward1 = []
    reward2 = []

    while epoch < 1000:
        t = 0
        episode1 = []
        episode2 = []
        epoch += 1
        print('reset')
        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        while True:
            print('a1')
            a1 = p1.get_action_value(s1)
            print('a2')
            a2 = p2.get_action_value(s2)
            print('step')
            s1_, s2_, r1, r2, done = env.step(a1, a2)
            print('add')
            episode1.append(r1)
            episode2.append(r2)
            s1 = s1_
            s2 = s2_
            t += 1

            print('next')
            if t > 200:
                print('>200b')
                reward1.append(sum(episode1))
                reward2.append(sum(episode2))
                break

            if done:
                print('doneb')
                reward1.append(sum(episode1))
                reward2.append(sum(episode2))
                break

    return reward1, reward2

if __name__ == '__main__':
    env = Game(8, 8)

    # policy
    p1 = PPOPolicy(is_training=False, model_path='model/1-3(150000)/1.ckpt', k=4)
    p2 = PPOPolicy(is_training=False, model_path='model/2-3(150000)/5.ckpt', k=4)

    epoch = 0
    reward1 = []
    reward2 = []

    while epoch < 1000:
        t = 0
        episode1 = []
        episode2 = []
        epoch += 1

        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        while True:

            a1,_ = p1.get_action_value(s1)
            a2,_ = p2.get_action_value(s2)

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            episode1.append(r1)
            episode2.append(r2)
            s1 = s1_
            s2 = s2_
            t += 1

            if t > 200:
                reward1.append(sum(episode1))
                reward2.append(sum(episode2))
                break

            if done:
                reward1.append(sum(episode1))
                reward2.append(sum(episode2))
                break

    print(reward1)
    print(reward2)
    print(sum(reward1))
    print(sum(reward2))
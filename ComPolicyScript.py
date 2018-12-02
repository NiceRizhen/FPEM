'''
  This file is a script file for Comparing different policy.
'''

import numpy as np
from GameEnv import Game
import matplotlib.pyplot as plt
from RandomPolicy import RandomPolicy
from PPOPolicy import PPOPolicy
from ExpertPolicy import ExpertPolicy

def run_with_w(pi_f, pi_g1, pi_g2, pi_g3):
    w = [0.2, 0.3, 0.5]
    policy = [pi_g1, pi_g2, pi_g3]

    t = 0
    epoch = 0
    reward = []
    episode = []
    g_pi = np.random.choice(policy, 1, p=w)[0]
    state = env.reset()
    while epoch < 100:

        if t % 10 == 0:
            g_pi = np.random.choice(policy,1,p=w)[0]

        a1, a2 = pi_f.choose_action(state)
        a3, a4 = g_pi.choose_action(state)
        state_, r1, r2 = env.step(a1, a2, a3, a4)
        episode.append(r2)
        state = state_
        t += 1
        if t % 300 == 0:
            reward.append(sum(episode))
            episode = []
            epoch += 1
            t = 0

    return reward



def run_with_policy(f_pi, g_pi):

    t = 0
    epoch = 0
    reward = []
    episode = []

    state = env.reset()
    while epoch < 100:

        a1, a2 = f_pi.choose_action(state)
        a3, a4 = g_pi.choose_action(state)
        state_, r1, r2 = env.step(a1, a2, a3, a4)
        episode.append(r2)
        state = state_
        t += 1
        if t % 300 == 0:
            reward.append(sum(episode))
            episode = []
            epoch += 1
            t = 0

    return reward

if __name__ == '__main__':
    env = Game(5, 5)

    # f policy
    #pi_f = PPOPolicy(is_training=False, model_path='model/policy_for_f/f1vsg1.ckpt')

    # g policy
    pi_ramdom = RandomPolicy()
    pi_expert = ExpertPolicy(0.15)
    pi_g1 = PPOPolicy(is_training=False, model_path='model/policy_for_g/g1vsr.ckpt')
    pi_g2 = PPOPolicy(is_training=False, model_path='model/policy_for_g/g2vsf1.ckpt')


    reward_random = run_with_policy(pi_ramdom, pi_ramdom)
    reward_expert = run_with_policy(pi_ramdom, pi_expert)
    reward_gvsfe = run_with_policy(pi_ramdom, pi_g1)
    reward_g2vsf1 = run_with_policy(pi_ramdom, pi_g2)


    x = range(len(reward_gvsfe))
    plt.plot(x, reward_random, label="Random")
    plt.plot(x, reward_expert, label="expert")
    plt.plot(x, reward_g2vsf1, label="g2")
    plt.plot(x, reward_gvsfe, label="g1")
   # plt.plot(x, reward_gvsf1, label="g2(f1)")
    #plt.plot(x, reward_mix, label="with W")
    plt.xlabel('Episode')
    plt.ylabel('Return( vs r)')
    plt.legend()
    plt.savefig('debug2.jpg')
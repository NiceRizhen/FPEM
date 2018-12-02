'''
  This file is a script file for Comparing different policy.
'''

from GameEnv import Game
import matplotlib.pyplot as plt
from RandomPolicy import RandomPolicy
from PPOPolicy import PPOPolicy
from ExpertPolicy import ExpertPolicy

def run_with_policy(f_pi, g_pi):

    t = 0
    epoch = 0
    reward = []
    episode = []

    state = env.reset()
    while epoch < 150:

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
    # pi_f = PPOPolicy(is_training=False, model_path='model/policy_for_f/fevse.ckpt')

    # g policy
    pi_ramdom = RandomPolicy()
    pi_expert = ExpertPolicy(0.15)
    pi_g1 = PPOPolicy(is_training=False, model_path='model/policy_for_g/g1vsr.ckpt')
    pi_g2 = PPOPolicy(is_training=False, model_path='model/policy_for_g/g2vsf1.ckpt')

    reward_random = run_with_policy(pi_ramdom ,pi_ramdom)
    reward_expert = run_with_policy(pi_ramdom, pi_expert)
    reward_gvsr = run_with_policy(pi_ramdom, pi_g1)
    reward_gvsf1 = run_with_policy(pi_ramdom, pi_g2)

    x = range(len(reward_random))
    plt.plot(x, reward_random, label="Random")
    plt.plot(x, reward_expert, label="Expert")
    plt.plot(x, reward_gvsr, label="g1(r)")
    plt.plot(x, reward_gvsf1, label="g2(f1)")
    plt.xlabel('Episode')
    plt.ylabel('Return( vs random)')
    plt.legend()
    plt.savefig('basepolicy-vs-r.jpg')
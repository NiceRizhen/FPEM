'''
  Entrance to Our Algorithm

  Runtime Dependencies:
    System         : Ubuntu 16.04
    python-version : 3.6.7
    tensorflow-gpu : 1.12.0
    IDE            : PyCharm

'''

from RandomPolicy import RandomPolicy
from GameEnv import Game
from PPOPolicy import PPOPolicy
import matplotlib.pyplot as plt

def wining_rate(vs_pi, tested_pi):

    loss = 0
    performance = []
    for i in range(100):
        win = 0
        for  epoch in range(100):
            s1, s2 = env.reset()
            while True:
                a1 = vs_pi.choose_action(s1)
                a2, v = tested_pi.get_action_value(s2)

                s1_,s2_,r1,r2,done = env.step(a1,a2)

                s1 = s1_
                s2 = s2_

                if done:
                    if r1+r2 == 2:
                        break
                    elif r2 == 1:
                        win += 1
                        break
                    else:
                        loss += 1
                        break
        performance.append(win)

    return performance

if __name__ == '__main__':
    env = Game(8,8)

    pi = RandomPolicy()
    pi1 = PPOPolicy(is_training=False, model_path='model/policy_for_g/pi1-20000.ckpt')
    pi2 = PPOPolicy(is_training=False, model_path='model/policy_for_g/pi1-40000.ckpt')
    pi3 = PPOPolicy(is_training=False, model_path='model/policy_for_g/pi1-60000.ckpt')

    per1 = wining_rate(pi, pi1)
    per2 = wining_rate(pi, pi2)
    per3 = wining_rate(pi, pi3)

    x = range(len(per1))
    plt.plot(x, per1, label="pi-20000")
    plt.plot(x, per2, label="pi-40000")
    plt.plot(x, per3, label="pi-60000")


    plt.xlabel('Episode')
    plt.ylabel('wining rate')
    plt.legend()
    plt.savefig('comparison.jpg')
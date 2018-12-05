'''
  Entrance to Our Algorithm

  Runtime Dependencies:
    System         : Ubuntu 16.04
    python-version : 3.6.7
    tensorflow-gpu : 1.12.0
    IDE            : PyCharm

'''
import math
from RandomPolicy import RandomPolicy
from GameEnv import Game
from PPOPolicy import PPOPolicy
import matplotlib.pyplot as plt

def performance(pi, pi_ppo):

    epoch = 0
    win = 0
    while epoch < 100:

        s, space = env.reset()
        s1 = s[0]
        s2 = s[1]
        epoch += 1

        while True:
            a1 = pi.choose_action(s1)
            a2 = pi_ppo.choose_action(s2)

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            s1 = s1_
            s2 = s2_

            if done:
                if r2==1 and r1!=1:
                    win += 1

                break

    print('win nums:{0}'.format(win))

if __name__ == '__main__':
    env = Game(8,8)

    pi = RandomPolicy()
    pi_ppo = PPOPolicy(log_path='model/logs/')

    epoch = 0
    while True:

        s, space= env.reset()
        s1 = s[0]
        s2 = s[1]
        epoch += 1
        t = 0

        while True:
            a1 = pi.choose_action(s1)
            a2, v = pi_ppo.get_action_value(s2)

            s1_,s2_,r1,r2,done = env.step(a1,a2)

            pi_ppo.save_transition(s2, a2, r2, v)

            s1 = s1_
            s2 = s2_

            t+= 1

            if done:
                if t < 2:
                    pi_ppo.empty_memory()
                    break
                pi_ppo.train(s2)
                break

        if epoch % 5000 == 0:
            performance(pi, pi_ppo)

        if epoch % 20000 == 0:
            pi_ppo.save_model('model/ppo-{0}'.format(epoch))
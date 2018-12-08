'''
  Entrance to Our Algorithm

  Runtime Dependencies:
    System         : Ubuntu 16.04
    python-version : 3.6.7
    tensorflow-gpu : 1.12.0
    IDE            : PyCharm

'''
import os
import math
from RandomPolicy import RandomPolicy
from GameEnv import Game
from PPOPolicy import PPOPolicy
import matplotlib.pyplot as plt

# def performance(pi1, pi2):
#
#     epoch = 0
#     win = 0
#     while epoch < 100:
#
#         s, space = env.reset()
#         s1 = s[0]
#         s2 = s[1]
#         t = 0
#         epoch += 1
#         pi_ppo.empty_memory()
#
#         while True:
#             a1 = pi1.choose_action(s1)
#             a2 = pi2.choose_action(s2)
#
#             s1_, s2_, r1, r2, done = env.step(a1, a2)
#
#             s1 = s1_
#             s2 = s2_
#             if t > 200:
#                 epoch -= 1
#                 break
#
#             if done:
#                 if r2==1 and r1!=1:
#                     win += 1
#
#                 break
#
#     print('win nums:{0}'.format(win))

if __name__ == '__main__':
    env = Game(8,8)

    pi = RandomPolicy()
    pi_ppo = PPOPolicy(log_path='model/logs/', k=4)

    epoch = 1
    all_t = 0
    t = 0

    while epoch < 400000:

        all_t += t
        t = 0
        epoch += 1

        s, space= env.reset()
        s1 = s[0]
        s2 = s[1]

        while True:
            a1 = pi.choose_action(s1)

            a2, v = pi_ppo.get_action_value(s2)

            s1_,s2_,r1,r2,done = env.step(a1,a2)

            pi_ppo.save_transition(s2, s2_, a2, r2, v, done)

            s1 = s1_
            s2 = s2_

            t += 1

            if t > 200:
                epoch -= 1
                pi_ppo.empty_traj_memory()
                break

            if done:
                break

        if epoch % 200 == 0:
            pi_ppo.train()


        if epoch % 40000 == 0:
            pi_ppo.save_model('model/ppo-{0}(4).ckpt'.format(epoch))
            os.mkdir('{0}:{1}'.format(epoch,all_t))

    os.system('poweroff')

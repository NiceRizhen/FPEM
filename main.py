'''
  Entrance to Our Algorithm

  Runtime Dependencies:
    System         : Ubuntu 16.04
    python-version : 3.6.7
    tensorflow-gpu : 1.12.0
    IDE            : PyCharm

'''

import os
import numpy as np
from MixPolicyBP import MixPolicy
from GameEnv import Game
from collections import deque

# params for mix policy training
K = 4
MAX_POLICY = 20

if __name__ == '__main__':
    env = Game(8,8)

    # set training params
    epoch = 1
    _train_num = 500
    who_is_training = 1
    _save_num = 40001
    _save_times = 9

    mix_policy1 = MixPolicy(max_policy=MAX_POLICY, log_path='model/logs1/', k=K)
    mix_policy2 = MixPolicy(max_policy=MAX_POLICY, log_path='model/logs2/', k=K)

    p1_state = deque(maxlen=4)
    p2_state = deque(maxlen=4)

    while epoch < 100:
        t = 0

        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        who_take_it = 0

        for i in range(4):
            zero_state = np.zeros([40])
            p1_state.append(zero_state)
            p2_state.append(zero_state)

        cur_policy_n = 1
        while True:

            p1_state.append(s1)
            p2_state.append(s2)

            state1 = np.array([])

            for obs in p1_state:
                state1 = np.hstack((state1, obs))

            state2 = np.array([])
            for obs in p2_state:
                state2 = np.hstack((state2, obs))

            if who_take_it == 0:
                state1 = np.hstack((state1, [0]))
                state2 = np.hstack((state2, [0]))
            elif who_take_it == 1:
                state1 = np.hstack((state1, [1]))
                state2 = np.hstack((state2, [0]))
            else:
                state1 = np.hstack((state1, [0]))
                state2 = np.hstack((state2, [1]))

            a1, v1 = mix_policy1.get_action(state1, cur_policy_n)
            a2, v2 = mix_policy2.get_action(state2, cur_policy_n)

            s1_, s2_, r1, r2, done, who_takes_it = env.step(a1, a2)

            mix_policy1.save_transition(state1, s1_, a1, r1, v1, done, t, who_takes_it, cur_policy_n)
            mix_policy2.save_transition(state2, s2_, a2, r2, v2, done, t, who_takes_it, cur_policy_n)

            s1 = s1_
            s2 = s2_
            t += 1

            if t > 100:

                mix_policy1.empty_traj_memory()
                mix_policy2.empty_traj_memory()

                break

            if done:
                epoch += 1
                break

        p1_state.clear()
        p2_state.clear()

        if epoch % 5 == 0:
            if who_is_training == 1:
                print('train1')
                mix_policy1.train(cur_policy_n)
                mix_policy1.empty_all_memory()
            else:
                print('train2')
                mix_policy2.train(cur_policy_n)
                mix_policy2.empty_all_memory()

            who_is_training = 1 if who_is_training == 2 else 2

            epoch += 1

        # if epoch % _save_num == 0:
        #     mix_policy1.save_model('model/mix/p1bp_2/{0}.ckpt'.format(_save_times))
        #     mix_policy2.save_model('model/mix/p2bp_2/{0}.ckpt'.format(_save_times))
        #     epoch += 1
        #     _save_times += 1


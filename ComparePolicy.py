from GameEnv import Game
from PPO.PPOIS import PPOIS
from collections import deque
import numpy as np
import random

_EVA_NUMS = 10000

should_act = ['up', 'down', 'left', 'right']
K = 1

if __name__ == '__main__':
    space = []

    env = Game(8,8)

    policy_set1 = []
    policy_set2 = []

    # for i in range(10):
    policy_set1.append(PPOIS(is_training=False, k=K, model_path='model/base_policy/1/0/latest.ckpt'))
        # policy_set2.append(PPOIS(is_training=False, k=4, model_path='model/base_policy/2/{0}/latest.ckpt'.format(i)))

    #mp1 = MixPolicy(max_policy=10, k=4, is_training=False, model_path='model/weight1/latest.ckpt')

    for p_for_2 in range(9, 10):

        p1_win = 0
        p2_win = 0
        reward_buffer1 = []
        reward_buffer2 = []
        epoch = 0
        all_chance = 0
        correct_time = 0

        while epoch < _EVA_NUMS:

            s, space = env.reset()
            s1 = s[0]
            s2 = s[1]

            t = 0
            traj_r1 = 0
            traj_r2 = 0

            p1_state = deque(maxlen=K)
            p2_state = deque(maxlen=K)

            for i in range(K):
                zero_state = np.zeros([61])
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

                #p1, mu, mp_v1 = mp1.get_policy(state1,9)

                # times[p1] = times[p1] + 1
                a1, _, _ = policy_set1[0].get_action_value(state1)
                #a2, _, _ = policy_set2[p_for_2].get_action_value(state2)
                a2 = random.randint(0,3)

                s1_, s2_, r1, r2, done, is_1_step = env.step(a1, a2)
                if is_1_step >=0:
                    all_chance += 1
                    ap,_ = policy_set1[0].get_act_prob(state1)
                    #print('best action:{0}, choose:{1}, act_prob:{2}, reward:{3}'.format(should_act[is_1_step], should_act[a1], ap, r1))
                    if is_1_step == a1:
                        correct_time += 1

                traj_r1 += r1
                traj_r2 += r2

                s1 = s1_
                s2 = s2_
                t += 1

                if t > 100:
                    if traj_r1 > traj_r2:
                        p1_win += 1
                    elif traj_r2 > traj_r1:
                        p2_win += 1
                    epoch += 1
                    reward_buffer1.append(traj_r1)
                    reward_buffer2.append(traj_r2)
                    break

                if done:
                    if traj_r1 > traj_r2:
                        p1_win += 1
                    elif traj_r2 > traj_r1:
                        p2_win += 1

                    reward_buffer1.append(traj_r1)
                    reward_buffer2.append(traj_r2)
                    epoch += 1
                    break
            if epoch % 100 == 0:
                print('rate', correct_time/all_chance)
                print('p1 win:{0}, p2 win:{1}'.format(p1_win, p2_win))

                correct_time = 0
                all_chance = 0

        print('------------------mix vs bp{0}----------------------'.format(p_for_2))
        print('mix\'s wining times: {0}'.format((p1_win - p2_win) / _EVA_NUMS))
        reward_buffer1 = np.array(reward_buffer1)
        print('mix\'s reward mean: {0}'.format(np.mean(reward_buffer1)))
        print('mix\'s reward variance: {0}'.format(np.std(reward_buffer1)))
        print()

    #times = [0] * 5
    for p_for_1 in range(10):

        for p_for_2 in range(p_for_1+1,10):

            p1_win = 0
            p2_win = 0
            reward_buffer1 = []
            reward_buffer2 = []
            epoch = 0

            while epoch < _EVA_NUMS:

                s, space = env.reset()
                s1 = s[0]
                s2 = s[1]

                t = 0
                traj_r1 = 0
                traj_r2 = 0

                p1_state = deque(maxlen=4)
                p2_state = deque(maxlen=4)

                for i in range(4):
                    zero_state = np.zeros([61])
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

                    #p1, mu, mp_v1 = mp1.get_policy(state1,4)

                    #times[p1] = times[p1] + 1
                    a1, _,_ = policy_set1[p_for_1].get_action_value(state1)
                    a2, _,_ = policy_set1[p_for_2].get_action_value(state2)

                    s1_,s2_,r1,r2,done = env.step(a1, a2)

                    traj_r1 += r1
                    traj_r2 += r2

                    s1 = s1_
                    s2 = s2_
                    t += 1

                    if t > 100:
                        if traj_r1 > traj_r2:
                            p1_win += 1
                        elif traj_r2 > traj_r1:
                            p2_win += 1
                        epoch += 1
                        reward_buffer1.append(traj_r1)
                        reward_buffer2.append(traj_r2)

                        break

                    if done:
                        if traj_r1 > traj_r2:
                            p1_win += 1
                        elif traj_r2 > traj_r1:
                            p2_win += 1

                        reward_buffer1.append(traj_r1)
                        reward_buffer2.append(traj_r2)
                        epoch += 1
                        break

            print('------------------bp{0} vs bp{1}----------------------'.format(p_for_1, p_for_2))
            print('bp{0}\'s wining times: {1}'.format(p_for_1, (p1_win-p2_win)/_EVA_NUMS))
            reward_buffer1 = np.array(reward_buffer1)
            print('bp{0}\'s reward mean: {1}'.format(p_for_1, np.mean(reward_buffer1)))
            print('bp{0}\'s reward variance: {1}'.format(p_for_1, np.std(reward_buffer1)))
            #print('choice times', [a/sum(times) for a in times])
            print()


    # epoch = 1
    #
    # standings1 = deque(maxlen=200)
    # standings2 = deque(maxlen=200)
    #
    # while epoch < 25000:
    #     s, space = env.reset()
    #     s1 = s[0]
    #     s2 = s[1]
    #
    #     t = 0
    #     traj_r1 = 0
    #     traj_r2 = 0
    #
    #     p1_state = deque(maxlen=4)
    #     p2_state = deque(maxlen=4)
    #
    #     for i in range(4):
    #         zero_state = np.zeros([41])
    #         p1_state.append(zero_state)
    #         p2_state.append(zero_state)
    #
    #     while True:
    #
    #         p1_state.append(s1)
    #         p2_state.append(s2)
    #
    #         state1 = np.array([])
    #         for obs in p1_state:
    #             state1 = np.hstack((state1, obs))
    #
    #         state2 = np.array([])
    #         for obs in p2_state:
    #             state2 = np.hstack((state2, obs))
    #
    #         a1 = policy1.get_action(state1)
    #         a2,v2 = policy2.get_action_value(state2)
    #
    #         s1_, s2_, r1, r2, done = env.step(a1, a2)
    #
    #         policy2.save_transition(state2, s2_, a2, r2, v2, done, t)
    #
    #         traj_r1 += r1
    #         traj_r2 += r2
    #
    #         s1 = s1_
    #         s2 = s2_
    #         t += 1
    #         if t > 100:
    #             policy2.empty_traj_memory()
    #             break
    #
    #         if done:
    #             if traj_r2 > traj_r1:
    #                 standings2.append(1)
    #             else:
    #                 standings2.append(0)
    #
    #             epoch += 1
    #
    #             break
    #     if epoch % 200 == 0:
    #         policy2.train()
    #         epoch += 1
    #
    # print('player2 wining rate',sum(standings2))
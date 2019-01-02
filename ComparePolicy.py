from GameEnv import Game
from PPOPolicy import PPOPolicy
from collections import deque
import numpy as np
from MixPolicyBP import MixPolicyBP

_EVA_NUMS = 10000

if __name__ == '__main__':
    space = []

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

    mp1 = MixPolicyBP(policy_n=5, is_training=False, k=4, model_path='model/mix/p1bp/12.ckpt')
    mp2 = MixPolicyBP(policy_n=5, is_training=False, k=4, model_path='model/mix/p2bp/12.ckpt')


    with open('com_result.txt', 'w') as file:
        # evaluate 5 base policies
        for p_in_1 in range(5):

            epoch = 0
            reward1 = []
            reward2 = []
            win_nums = 0
            draw_nums = 0
            lose_nums = 0

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

                    index2 = mp2.get_policy(state2)[0]

                    a1 = ps1[p_in_1].get_action_full_state(state1)
                    a2 = ps2[index2].get_action_full_state(state2)

                    s1_,s2_,r1,r2,done = env.step(a1,a2)

                    traj_r1 += r1
                    traj_r2 += r2

                    s1 = s1_
                    s2 = s2_
                    t += 1
                    if t > 100:
                        break

                    if done:
                        if traj_r1 > traj_r2:
                            win_nums += 1
                        elif traj_r1 == traj_r2:
                            draw_nums += 1
                        else:
                            lose_nums += 1
                        reward1.append(traj_r1)
                        reward2.append(traj_r2)
                        epoch += 1

                        break

            # record sample data
            file.write("----------------base policy {0} vs mix policy---------------\n".format(p_in_1+1))
            reward1 = np.array(reward1)
            reward2 = np.array(reward2)

            reward_means = np.mean(reward1)
            reward_var = np.var(reward1)
            file.write("wining rate : {0}\n".format(win_nums/_EVA_NUMS))
            file.write("draw rate : {0}\n".format(draw_nums/_EVA_NUMS))
            file.write("lose rate : {0}\n".format(lose_nums/_EVA_NUMS))
            file.write("reward sum : {0}\n".format(np.sum(reward1)))
            file.write("reward mean : {0}\n".format(np.mean(reward1)))
            file.write("reward variance : {0}\n".format(np.var(reward1)))

            file.write("opponent's reward sum : {0}\n".format(np.sum(reward2)))
            file.write("opponent's reward mean : {0}\n".format(np.mean(reward2)))
            file.write("opponent's reward variance : {0}\n\n".format(np.var(reward2)))

        # evaluate mix policy
        epoch = 0
        reward1 = []
        reward2 = []
        win_nums = 0
        draw_nums = 0
        lose_nums = 0

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

                index1 = mp1.get_policy(state1)[0]
                index2 = mp2.get_policy(state2)[0]

                a1 = ps1[index1].get_action_full_state(state1)
                a2 = ps2[index2].get_action_full_state(state2)

                s1_, s2_, r1, r2, done = env.step(a1, a2)

                traj_r1 += r1
                traj_r2 += r2

                s1 = s1_
                s2 = s2_
                t += 1
                if t > 100:
                    break

                if done:
                    if traj_r1 > traj_r2:
                        win_nums += 1
                    elif traj_r1 == traj_r2:
                        draw_nums += 1
                    else:
                        lose_nums += 1
                    reward1.append(traj_r1)
                    reward2.append(traj_r2)

                    epoch += 1
                    break

        file.write("----------------mix policy vs opponent's mix policy---------------\n")
        reward1 = np.array(reward1)
        reward2 = np.array(reward2)

        file.write("wining rate : {0}\n".format(win_nums / _EVA_NUMS))
        file.write("draw rate : {0}\n".format(draw_nums / _EVA_NUMS))
        file.write("lose rate : {0}\n".format(lose_nums / _EVA_NUMS))
        file.write("reward sum : {0}\n".format(np.sum(reward1)))
        file.write("reward mean : {0}\n".format(np.mean(reward1)))
        file.write("reward variance : {0}\n".format(np.var(reward1)))

        file.write("opponent's reward sum : {0}\n".format(np.sum(reward2)))
        file.write("opponent's reward mean : {0}\n".format(np.mean(reward2)))
        file.write("opponent's reward variance : {0}\n\n".format(np.var(reward2)))
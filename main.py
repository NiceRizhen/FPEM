'''
  Entrance to Our Algorithm

  Runtime Dependencies:
    System         : Ubuntu 16.04
    python-version : 3.6.7
    tensorflow-gpu : 1.12.0
    IDE            : PyCharm

'''

# model path : model/base_policy/1/{0}/latest.ckpt

import copy
import numpy as np
from MixPolicyBP import MixPolicy
from GameEnv import Game
from collections import deque
from PPOIS import PPOIS
from multiprocessing import Process, Manager, Lock

# params for mix policy training
K = 1        # combine history
MAX_POLICY = 2
MAX_EPOCH = 80
_END_THREHOLD = 0.03
_GAMMA = 0.99
_LAMBDA = 0.95
_PROCESS_NUM = 4
_SAMPLE_EPOCH = 100
_NEED_MODEL = 0

# worker for process to optimize base policy and mix policy
def optimize_worker(steps, bp_steps, mp_steps, cur_policy, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mu_probs, mp_mu_probs, mp_policys, mp_gaes, mp_v_preds_next):

    if who_is_training == 1:
        base_policy1 = PPOIS(is_continuing=True, k=K, log_path='log/log1/{0}/'.format(cur_policy),
                                 model_path='model/base_policy/1/{0}/latest.ckpt'.format(cur_policy))

        base_policy1.n_training = bp_steps
        base_policy1.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next, mu_probs)
        base_policy1.save_model()

        mp1 = MixPolicy(max_policy=MAX_POLICY, is_continuing=True, log_path='log/weight1/',
                        k=K, model_path='model/weight1/latest.ckpt')

        mp1.n_training = mp_steps
        mp1.train(obses, mp_policys, mp_gaes, rewards, mp_v_preds_next, cur_policy, mp_mu_probs)
        mp1.save_model()

        steps.append(base_policy1.n_training)
        steps.append(mp1.n_training)

    else:
        base_policy2 = PPOIS(is_continuing=True, k=K, log_path='log/log2/{0}/'.format(cur_policy),
                                 model_path='model/base_policy/2/{0}/latest.ckpt'.format(cur_policy))

        base_policy2.n_training = bp_steps
        base_policy2.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next, mu_probs)
        base_policy2.save_model()

        steps.append(base_policy2.n_training)

def optimize_worker_0(steps, bp_steps, who_is_training, obses, bp_actions, bp_gaes, rewards, bp_v_preds_next, mu_probs):

    if who_is_training == 1:

        base_policy1 = PPOIS(is_continuing=True, k=K, log_path='log/log1/0/',
                                 model_path='model/base_policy/1/0/latest.ckpt')

        base_policy1.n_training = bp_steps
        base_policy1.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next, mu_probs)
        base_policy1.save_model()

        steps.append(base_policy1.n_training)

    else:
        base_policy2 = PPOIS(is_continuing=True, k=K, log_path='log/log2/0',
                                 model_path='model/base_policy/2/0/latest.ckpt')

        base_policy2.n_training = bp_steps
        base_policy2.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next, mu_probs)
        base_policy2.save_model()

        steps.append(base_policy2.n_training)

def optimize_by_childprocess(bp_steps,
                             mp_steps,
                             cur_policy_n,
                             who_is_training,
                             obses,
                             bp_actions,
                             bp_gaes,
                             rewards,
                             bp_v_preds_next,
                             mu_probs,
                             mp_mu_probs,
                             mp_policys,
                             mp_gaes,
                             mp_v_preds_next):

    manager = Manager()
    steps = manager.list([])

    if cur_policy_n == 0:
        p = Process(target=optimize_worker_0,
                    args=(steps,
                          bp_steps,
                          who_is_training,
                          obses,
                          bp_actions,
                          bp_gaes,
                          rewards,
                          bp_v_preds_next,
                          mu_probs,))
        p.start()
        p.join()

        return steps[0]

    else:

        p = Process(target=optimize_worker,
                    args=(steps,
                          bp_steps,
                          mp_steps,
                          cur_policy_n,
                          who_is_training,
                          obses, bp_actions,
                          bp_gaes,
                          rewards,
                          bp_v_preds_next,
                          mu_probs,
                          mp_mu_probs,
                          mp_policys,
                          mp_gaes,
                          mp_v_preds_next,))
        p.start()
        p.join()

        if who_is_training == 1:
            return  steps[0], steps[1]
        else:
            return  steps[0]

def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + _GAMMA * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + _GAMMA * _LAMBDA * gaes[t + 1]
    return gaes

def p2_worker(cur_policy, wining_rate, mutex,
           obses, rewards, mu_prob, bp_actions, bp_gaes, bp_v_preds_next):

    env = Game(8, 8)

    policy_set1 = []
    policy_set2 = []

    if cur_policy == 0:
        policy_set1.append(PPOIS(is_training=False, k=K, model_path='model/base_policy/1/0/latest.ckpt'))
        policy_set2.append(
            PPOIS(is_training=False, k=K,
                                  model_path='model/base_policy/2/0/latest.ckpt')
        )

    # after first iteration
    else:
        for i in range(cur_policy + 1):
            policy_set1.append(
                PPOIS(is_training=False, k=K, model_path='model/base_policy/1/{0}/latest.ckpt'.format(i))
            )

        policy_set2.append(
            PPOIS(is_training=False, k=K, model_path='model/base_policy/2/{0}/latest.ckpt'.format(cur_policy))
        )

    mp1 = MixPolicy(max_policy=MAX_POLICY, is_training=False, k=K, model_path='model/weight1/latest.ckpt')

    epoch = 0
    win_num = 0
    lose_num = 0
    p1_state = deque(maxlen=K)
    p2_state = deque(maxlen=K)

    while epoch < _SAMPLE_EPOCH:

        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        # to record timesteps
        t = 0

        # to record winning nums
        traj_r1 = 0
        traj_r2 = 0

        # memory buffer for this trajectory
        traj_obs = []
        traj_reward = []
        traj_bp_actions = []
        traj_bp_v = []
        traj_mu_prob = []

        # initial state
        for i in range(K):
            zero_state = np.zeros([61])
            p1_state.append(zero_state)
            p2_state.append(zero_state)

        # a trajectory with max steps=100
        while True:

            p1_state.append(s1)
            p2_state.append(s2)

            state1 = np.array([])
            for obs in p1_state:
                state1 = np.hstack((state1, obs))

            state2 = np.array([])
            for obs in p2_state:
                state2 = np.hstack((state2, obs))

            p1, _,_ = mp1.get_policy(state1, cur_policy)
            a1, _, _ = policy_set1[p1].get_action_value(state1)

            a2, mu, bp_v2 = policy_set2[0].get_action_value(state2)
            s1_, s2_, r1, r2, done = env.step(a1, a2)

            traj_r1 += r1
            traj_r2 += r2

            t += 1

            if t > 100:

                if traj_r2 > traj_r1:
                    win_num += 1
                elif traj_r1 > traj_r2:
                    lose_num += 1

                traj_obs.append(state2)
                traj_reward.append(r2)
                traj_bp_actions.append(a2)
                traj_bp_v.append(bp_v2)
                traj_mu_prob.append(mu)

                # add data to memory buffer
                mutex.acquire()

                obses.append(traj_obs)
                bp_actions.append(traj_bp_actions)
                rewards.append(traj_reward)
                mu_prob.append(traj_mu_prob)

                traj_bp_v_next = traj_bp_v[1:] + [traj_bp_v[-1]]

                bp_v_preds_next.append(traj_bp_v_next)

                traj_bp_gaes = get_gaes(traj_reward, traj_bp_v, traj_bp_v_next)
                traj_bp_gaes = np.array(traj_bp_gaes).astype(dtype=np.float32)
                traj_bp_gaes = (traj_bp_gaes - traj_bp_gaes.mean()) / traj_bp_gaes.std()
                traj_bp_gaes = np.squeeze(traj_bp_gaes).tolist()
                bp_gaes.append(traj_bp_gaes)

                mutex.release()
                epoch += 1
                break

            elif done:

                if t > 1:

                    if traj_r2 > traj_r1:
                        win_num += 1
                    elif traj_r1 > traj_r2:
                        lose_num += 1

                    traj_obs.append(state2)
                    traj_reward.append(r2)
                    traj_bp_actions.append(a2)
                    traj_bp_v.append(bp_v2)
                    traj_mu_prob.append(mu)

                    # add data to memory buffer
                    mutex.acquire()

                    obses.append(traj_obs)
                    bp_actions.append(traj_bp_actions)
                    rewards.append(traj_reward)
                    mu_prob.append(traj_mu_prob)

                    traj_bp_v_next = traj_bp_v[1:] + [traj_bp_v[-1]]

                    bp_v_preds_next.append(traj_bp_v_next)

                    traj_bp_gaes = get_gaes(traj_reward, traj_bp_v, traj_bp_v_next)
                    traj_bp_gaes = np.array(traj_bp_gaes).astype(dtype=np.float32)
                    traj_bp_gaes = (traj_bp_gaes - traj_bp_gaes.mean()) / traj_bp_gaes.std()
                    traj_bp_gaes = np.squeeze(traj_bp_gaes).tolist()
                    bp_gaes.append(traj_bp_gaes)

                    mutex.release()
                    epoch += 1
                    break

                # this trajectory's length = 1
                else:
                    break

            else:
                traj_obs.append(state2)
                traj_reward.append(r2)
                traj_bp_actions.append(a2)
                traj_bp_v.append(bp_v2)
                traj_mu_prob.append(mu)

            s1 = s1_
            s2 = s2_

    wining_rate.append((win_num - lose_num) / _SAMPLE_EPOCH)

# a worker to get samples
def p1_worker(cur_policy, wining_rate, mutex,
           obses, rewards, mu_probs, mp_mu_probs,
           bp_actions, bp_gaes, bp_v_preds_next,
           mp_policys, mp_gaes, mp_v_preds_next):

    '''
    :param p1_index: policy that can be excuted
    :param p2_index:
    :param obs: obs for both policy and mix_policy
    :param rewards:
    :param bp_action:  for base policy
    :param mp_policy:  for mix policy
    :param wining_rate:
    :mutex: a process lock to protect sample order
    :return:
    '''

    env = Game(8,8)

    policy_set1 = []
    policy_set2 = []

    if cur_policy == 0:
        policy_set1.append(PPOIS(is_training=False, k=K, model_path='model/base_policy/1/0/latest.ckpt'))
        policy_set2.append(PPOIS(is_training=False, k=K, model_path='model/base_policy/2/0/latest.ckpt'))

    else:
        for i in range(cur_policy + 1):
            basepolicy = PPOIS(is_training=False, k=K, model_path='model/base_policy/1/{0}/latest.ckpt'.format(i))
            policy_set1.append(basepolicy)

        for i in range(cur_policy):
            basepolicy = PPOIS(is_training=False, k=K, model_path='model/base_policy/2/{0}/latest.ckpt'.format(i))
            policy_set2.append(basepolicy)

    mp1 = MixPolicy(max_policy=MAX_POLICY, is_training=False, k=K, model_path='model/weight1/latest.ckpt')

    epoch = 0
    p2_index = 0
    win_num = 0
    lose_num = 0
    p1_state = deque(maxlen=K)
    p2_state = deque(maxlen=K)

    while epoch < _SAMPLE_EPOCH:

        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        # to record timesteps
        t = 0

        # to record winning nums
        traj_r1 = 0
        traj_r2 = 0

        # memory buffer for this trajectory
        traj_obs = []
        traj_reward = []
        traj_pi_prob = []
        traj_mu_prob = []
        traj_mp_mu_prob = []

        traj_bp_actions = []
        traj_bp_v = []

        traj_mp_policy = []
        traj_mp_v = []

        # initial state
        for i in range(K):
            zero_state = np.zeros([61])
            p1_state.append(zero_state)
            p2_state.append(zero_state)

        # a trajectory with max steps=100
        while True:

            p1_state.append(s1)
            p2_state.append(s2)

            state1 = np.array([])
            for obs in p1_state:
                state1 = np.hstack((state1, obs))

            state2 = np.array([])
            for obs in p2_state:
                state2 = np.hstack((state2, obs))

            p1, mp_mu_prob, mp_v1 = mp1.get_policy(state1, cur_policy)
            a1, mu_prob, _ = policy_set1[p1].get_action_value(state1)
            pi_prob, bp_v1 = policy_set1[cur_policy].get_act_prob(state1)
            pi_prob = pi_prob[a1]

            a2,_, bp_v2 = policy_set2[p2_index].get_action_value(state2)

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            traj_r1+=r1
            traj_r2+=r2

            t += 1

            if t > 100:

                if traj_r1 > traj_r2:
                    win_num += 1
                elif traj_r2 > traj_r1:
                    lose_num += 1

                traj_obs.append(state1)
                traj_reward.append(r1)
                traj_pi_prob.append(round(pi_prob,4))
                traj_mu_prob.append(round(mu_prob,4))
                traj_mp_mu_prob.append(mp_mu_prob)

                traj_bp_actions.append(a1)
                traj_bp_v.append(bp_v1)

                traj_mp_policy.append(p1)
                traj_mp_v.append(mp_v1)

                #is_ratios = [pi/mu for pi, mu in zip(traj_pi_prob, traj_mu_prob)]

                # add data to memory buffer
                mutex.acquire()

                obses.append(traj_obs)
                mu_probs.append(traj_mu_prob)
                bp_actions.append(traj_bp_actions)
                mp_policys.append(traj_mp_policy)
                rewards.append(traj_reward)
                mp_mu_probs.append(traj_mp_mu_prob)

                traj_bp_v_next = traj_bp_v[1:] + [traj_bp_v[-1]]
                traj_mp_v_next = traj_mp_v[1:] + [traj_mp_v[-1]]

                bp_v_preds_next.append(traj_bp_v_next)
                mp_v_preds_next.append(traj_mp_v_next)

                traj_bp_gaes = get_gaes(traj_reward, traj_bp_v, traj_bp_v_next)
                traj_bp_gaes = np.array(traj_bp_gaes).astype(dtype=np.float32)
                traj_bp_gaes = (traj_bp_gaes - traj_bp_gaes.mean()) / traj_bp_gaes.std()
                traj_bp_gaes = np.squeeze(traj_bp_gaes).tolist()
                bp_gaes.append(traj_bp_gaes)

                traj_mp_gaes = get_gaes(traj_reward, traj_mp_v, traj_mp_v_next)
                traj_mp_gaes = np.array(traj_mp_gaes).astype(dtype=np.float32)
                traj_mp_gaes = (traj_mp_gaes - traj_mp_gaes.mean()) / traj_mp_gaes.std()
                traj_mp_gaes = np.squeeze(traj_mp_gaes).tolist()
                mp_gaes.append(traj_mp_gaes)

                mutex.release()
                epoch += 1
                break

            elif done:

                if t > 1:

                    if traj_r1 > traj_r2:
                        win_num += 1
                    elif traj_r2 > traj_r1:
                        lose_num += 1
                    traj_obs.append(state1)
                    traj_reward.append(r1)
                    traj_pi_prob.append(round(pi_prob, 4))
                    traj_mu_prob.append(round(mu_prob, 4))
                    traj_mp_mu_prob.append(mp_mu_prob)

                    traj_bp_actions.append(a1)
                    traj_bp_v.append(bp_v1)

                    traj_mp_policy.append(p1)
                    traj_mp_v.append(mp_v1)

                    # add data to memory buffer
                    mutex.acquire()

                    obses.append(traj_obs)
                    mu_probs.append(traj_mu_prob)
                    bp_actions.append(traj_bp_actions)
                    mp_policys.append(traj_mp_policy)
                    rewards.append(traj_reward)
                    mp_mu_probs.append(traj_mp_mu_prob)

                    traj_bp_v_next = traj_bp_v[1:] + [traj_bp_v[-1]]
                    traj_mp_v_next = traj_mp_v[1:] + [traj_mp_v[-1]]

                    bp_v_preds_next.append(traj_bp_v_next)
                    mp_v_preds_next.append(traj_mp_v_next)

                    traj_bp_gaes = get_gaes(traj_reward, traj_bp_v, traj_bp_v_next)
                    traj_bp_gaes = np.array(traj_bp_gaes).astype(dtype=np.float32)
                    traj_bp_gaes = (traj_bp_gaes - traj_bp_gaes.mean()) / traj_bp_gaes.std()
                    traj_bp_gaes = np.squeeze(traj_bp_gaes).tolist()
                    bp_gaes.append(traj_bp_gaes)

                    traj_mp_gaes = get_gaes(traj_reward, traj_mp_v, traj_mp_v_next)
                    traj_mp_gaes = np.array(traj_mp_gaes).astype(dtype=np.float32)
                    traj_mp_gaes = (traj_mp_gaes - traj_mp_gaes.mean()) / traj_mp_gaes.std()
                    traj_mp_gaes = np.squeeze(traj_mp_gaes).tolist()
                    mp_gaes.append(traj_mp_gaes)

                    mutex.release()
                    epoch += 1
                    break

                # this trajectory's length = 1
                else:
                    break
            else:

                traj_obs.append(state1)
                traj_reward.append(r1)
                traj_pi_prob.append(round(pi_prob, 4))
                traj_mu_prob.append(round(mu_prob, 4))
                traj_mp_mu_prob.append(mp_mu_prob)

                traj_bp_actions.append(a1)
                traj_bp_v.append(bp_v1)

                traj_mp_policy.append(p1)
                traj_mp_v.append(mp_v1)

            s1 = s1_
            s2 = s2_

        p2_index += 1
        if p2_index >= cur_policy:
            p2_index = 0

    wining_rate.append((win_num-lose_num)/_SAMPLE_EPOCH)

def training_process(mp1_steps, mp2_steps, cur_policy_n, who_is_training):

    # first iteration
    manager = Manager()

    if cur_policy_n == 0:

        if who_is_training == 1:
            epoch = 0
            end_flag = False

            bp_steps = 0

            while epoch < 80:

                wining_rate = manager.list([])
                mutex = Lock()
                obses = manager.list([])
                rewards = manager.list([])
                bp_actions = manager.list([])
                bp_gaes = manager.list([])
                bp_v_preds_next = manager.list([])
                mp_policys = manager.list([])
                mp_gaes = manager.list([])
                mp_v_preds_next = manager.list([])
                mu_probs = manager.list([])
                mp_mu_probs = manager.list([])

                process_list = []

                # sample processes
                for i in range(_PROCESS_NUM):
                    p = Process(target=p1_worker, args=(0,
                                                        wining_rate,
                                                        mutex,
                                                        obses,
                                                        rewards,
                                                        mu_probs,
                                                        mp_mu_probs,
                                                        bp_actions,
                                                        bp_gaes,
                                                        bp_v_preds_next,
                                                        mp_policys,
                                                        mp_gaes,
                                                        mp_v_preds_next,))
                    process_list.append(p)

                for p in process_list:
                    p.start()

                for p in process_list:
                    p.join()

                wr = sum(wining_rate)/_PROCESS_NUM

                epoch += 1
                print('player1\'s bp0 vs player1\'s bp0:{0} advantage, player{1} is training!'.format(wr, who_is_training))

                # if wr > _END_THREHOLD:
                #    if end_flag:
                #        print('break when epoch = {0}'.format(epoch))
                #        break
                #    else:
                #        end_flag = True
                # else:
                #    end_flag = False

                bp_steps = optimize_by_childprocess(bp_steps,
                                                    0,
                                                    0,
                                                    1,
                                                    obses,
                                                    bp_actions,
                                                    bp_gaes,
                                                    rewards,
                                                    bp_v_preds_next,
                                                    mu_probs,
                                                    mp_mu_probs,
                                                    mp_policys,
                                                    mp_gaes,
                                                    mp_v_preds_next)

        # first iteration, player2
        else:
            epoch = 0
            end_flag = False

            bp_steps = 0

            while epoch < MAX_EPOCH:

                wining_rate = manager.list([])
                mutex = Lock()
                obses = manager.list([])
                rewards = manager.list([])
                bp_actions = manager.list([])
                bp_gaes = manager.list([])
                bp_v_preds_next = manager.list([])
                mu_probs = manager.list([])

                process_list = []

                # sample processes
                for i in range(_PROCESS_NUM):
                    p = Process(target=p2_worker, args=(0,
                                                        wining_rate,
                                                        mutex,
                                                        obses,
                                                        rewards,
                                                        mu_probs,
                                                        bp_actions,
                                                        bp_gaes,
                                                        bp_v_preds_next,))
                    process_list.append(p)

                for p in process_list:
                    p.start()

                for p in process_list:
                    p.join()

                wr = sum(wining_rate) / _PROCESS_NUM

                epoch += 1
                print('player1\'s bp0 vs player1\'s bp0:{0} advantage, player{1} is training!'.format(wr, who_is_training))

                if wr > _END_THREHOLD:
                    if end_flag:
                        print('break when epoch = {0}'.format(epoch))
                        break
                    else:
                        end_flag = True
                else:
                    end_flag = False

                bp_steps = optimize_by_childprocess(bp_steps,
                                                    0,
                                                    0,
                                                    2,
                                                    obses,
                                                    bp_actions,
                                                    bp_gaes,
                                                    rewards,
                                                    bp_v_preds_next,
                                                    mu_probs,
                                                    None,
                                                    None,
                                                    None,
                                                    None)

        return 0

    # after first iteration
    else:
        # player1 is training
        if who_is_training == 1:

            bp_steps = 0
            mp_steps = mp1_steps

            # player1 trains a new base policy to defeat every policy of player2
            # to defeat n player2's base policy

            epoch = 0
            end_flag = False

            while epoch < MAX_EPOCH:

                wining_rate = manager.list([])
                mutex = Lock()
                obses = manager.list([])
                rewards = manager.list([])
                mu_probs = manager.list([])
                bp_actions = manager.list([])
                bp_gaes = manager.list([])
                bp_v_preds_next = manager.list([])
                mp_mu_probs = manager.list([])
                mp_policys = manager.list([])
                mp_gaes = manager.list([])
                mp_v_preds_next = manager.list([])

                process_list = []
                # sampling processes
                for i in range(_PROCESS_NUM):
                    p = Process(target=p1_worker, args=(cur_policy_n,
                                                    wining_rate,
                                                    mutex,
                                                    obses,
                                                    rewards,
                                                    mu_probs,
                                                    mp_mu_probs,
                                                    bp_actions,
                                                    bp_gaes,
                                                    bp_v_preds_next,
                                                    mp_policys,
                                                    mp_gaes,
                                                    mp_v_preds_next,))
                    process_list.append(p)

                for p in process_list:
                    p.start()

                for p in process_list:
                    p.join()

                wr = sum(wining_rate) / _PROCESS_NUM

                epoch += 1
                print('player1 mix bp0-{0} vs player2:{1} advantage, player1 is training!'.format(cur_policy_n, wr))

                if wr > _END_THREHOLD:
                    if end_flag:
                        print('break when epoch = {0}'.format(epoch))
                        break
                    else:
                        end_flag = True
                else:
                    end_flag = False

                bp_steps, mp_steps = optimize_by_childprocess(bp_steps,
                                                              mp_steps,
                                                              cur_policy_n,
                                                              1,
                                                              obses,
                                                              bp_actions,
                                                              bp_gaes,
                                                              rewards,
                                                              bp_v_preds_next,
                                                              mu_probs,
                                                              mp_mu_probs,
                                                              mp_policys,
                                                              mp_gaes,
                                                              mp_v_preds_next)

        # player2 is training
        else:

            bp_steps = 0
            mp_steps = mp2_steps

            epoch = 0
            end_flag = False

            while epoch < MAX_EPOCH:

                wining_rate = manager.list([])
                mutex = Lock()
                obses = manager.list([])
                rewards = manager.list([])
                bp_actions = manager.list([])
                bp_gaes = manager.list([])
                bp_v_preds_next = manager.list([])
                mu_probs = manager.list([])

                process_list = []

                # sampling processes
                for i in range(_PROCESS_NUM):
                    p = Process(target=p2_worker, args=(cur_policy_n,
                                                        wining_rate,
                                                        mutex,
                                                        obses,
                                                        rewards,
                                                        mu_probs,
                                                        bp_actions,
                                                        bp_gaes,
                                                        bp_v_preds_next,))
                    process_list.append(p)

                for p in process_list:
                    p.start()

                for p in process_list:
                    p.join()

                wr = sum(wining_rate) / _PROCESS_NUM
                epoch += 1
                print('player1 mix bp0-{0} vs player2\'s bp{0}:{1} advantage, player2 is training!'.format(cur_policy_n, wr))

                if wr > _END_THREHOLD:
                    if end_flag:
                        print('break when epoch = {0}'.format(epoch))
                        break
                    else:
                        end_flag = True
                else:
                    end_flag = False

                bp_steps  = optimize_by_childprocess(bp_steps,
                                                     0,
                                                     cur_policy_n,
                                                     2,
                                                     obses,
                                                     bp_actions,
                                                     bp_gaes,
                                                     rewards,
                                                     bp_v_preds_next,
                                                     mu_probs,
                                                     None,
                                                     None,
                                                     None,
                                                     None)

        return mp_steps

if __name__ == '__main__':
    flag = _NEED_MODEL

    if flag == 1:
        #initialize weight
        mix_policy1 = MixPolicy(max_policy=MAX_POLICY, log_path='log/weight1/', k=K, model_path='model/weight1/latest.ckpt')

        #initialize base policy
        policy_set1 = []
        policy_set2 = []

        for i in range(MAX_POLICY):
           policy_set1.append(
               PPOIS(k=K,model_path='model/base_policy/1/{0}/latest.ckpt'.format(i), log_path='log/log1/{0}/'.format(i))
           )
           policy_set2.append(
               PPOIS(k=K,model_path='model/base_policy/2/{0}/latest.ckpt'.format(i), log_path='log/log2/{0}/'.format(i))
           )

        for i in range(MAX_POLICY):
           policy_set1[i].save_model()
           policy_set2[i].save_model()

        mix_policy1.save_model()

    # training times for weight1 and 2
    mp1_n_training = 0
    mp2_n_training = 0

    for cur_policy_n in range(MAX_POLICY):

        print('---------iteration {0}  starts-------------'.format(cur_policy_n))
        mp1_n_training = training_process(mp1_n_training, mp2_n_training, cur_policy_n, who_is_training=1)

        print('player1 ends training base policy'.format(cur_policy_n))

        mp2_n_training = training_process(mp1_n_training, mp2_n_training, cur_policy_n, who_is_training=2)
        print('---------iteration {0}  ends---------------'.format(cur_policy_n))

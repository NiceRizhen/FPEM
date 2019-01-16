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
from PPOPolicy import PPOPolicy
from multiprocessing import Process, Manager, Lock

# params for mix policy training
K = 4        # combine history
MAX_POLICY = 10
MAX_EPOCH = 30
_END_THREHOLD = 0.05
_FIRST_END_THREHOLD = 0.1
_GAMMA = 0.99
_LAMBDA = 0.95
_PROCESS_NUM = 10
_SAMPLE_EPOCH = 100
_NEED_MODEL = True

# worker for process to optimize base policy and mix policy
def optimize_worker(steps, bp_steps, mp_steps, cur_policy_n, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next):

    if who_is_training == 1:
        base_policy1 = PPOPolicy(is_continuing=True, k=K, log_path='log/log1/{0}/'.format(cur_policy_n),
                                 model_path='model/base_policy/1/{0}/latest.ckpt'.format(cur_policy_n))

        base_policy1.n_training = bp_steps
        base_policy1.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy1.save_model()

        mp1 = MixPolicy(max_policy=MAX_POLICY, is_continuing=True, log_path='log/weight1/',
                        k=K, model_path='model/weight1/latest.ckpt')

        mp1.n_training = mp_steps
        mp1.train(obses, mp_policys, mp_gaes,rewards,mp_v_preds_next, cur_policy_n)
        mp1.save_model()

        steps.append(base_policy1.n_training)
        steps.append(mp1.n_training)

    else:
        base_policy2 = PPOPolicy(is_continuing=True, k=K, log_path='log/log2/{0}/'.format(cur_policy_n),
                                 model_path='model/base_policy/2/{0}/latest.ckpt'.format(cur_policy_n))

        base_policy2.n_training = bp_steps
        base_policy2.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy2.save_model()

        mp2 = MixPolicy(max_policy=MAX_POLICY, is_continuing=True, log_path='log/weight2/',
                        k=K, model_path='model/weight2/latest.ckpt')

        mp2.n_training = mp_steps
        mp2.train(obses, mp_policys, mp_gaes,rewards,mp_v_preds_next, cur_policy_n)
        mp2.save_model()

        steps.append(base_policy2.n_training)
        steps.append(mp2.n_training)

def optimize_worker_0(steps, bp_steps, who_is_training, obses, bp_actions, bp_gaes, rewards, bp_v_preds_next):

    if who_is_training == 1:

        base_policy1 = PPOPolicy(is_continuing=True, k=K, log_path='log/log1/0/',
                                 model_path='model/base_policy/1/0/latest.ckpt')

        base_policy1.n_training = bp_steps
        base_policy1.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy1.save_model()

        steps.append(base_policy1.n_training)
    else:
        base_policy2 = PPOPolicy(is_continuing=True, k=K, log_path='log/log2/0',
                                 model_path='model/base_policy/2/0/latest.ckpt')

        base_policy2.n_training = bp_steps
        base_policy2.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy2.save_model()
        steps.append(base_policy2.n_training)

def optimize_by_childprocess(bp_steps, mp_steps, cur_policy_n, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next):

    manager = Manager()
    steps = manager.list([])

    if cur_policy_n == 0:
        p = Process(target=optimize_worker_0,
                    args=(steps, bp_steps, who_is_training, obses, bp_actions, bp_gaes, rewards, bp_v_preds_next,))
        p.start()
        p.join()

        return steps[0]

    else:

        p = Process(target=optimize_worker,
                    args=(steps, bp_steps, mp_steps, cur_policy_n, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next,))
        p.start()
        p.join()

        return  steps[0], steps[1]

# pi prob is target policy while mu_prob is sampling policy
def bp_get_gaes(rewards, v_preds, v_preds_next, pi_prob, mu_prob):
    deltas = [r_t + _GAMMA * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)

    running_ratios = 1

    for t in reversed(range(len(gaes))):

        running_ratios *= pi_prob[t]/mu_prob[t]
        gaes[t] = gaes[t] * running_ratios

    # importance sampling ratios
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + _GAMMA * _LAMBDA * gaes[t + 1]

    return gaes

def mp_get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + _GAMMA * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + _GAMMA * _LAMBDA * gaes[t + 1]
    return gaes

# a worker to get samples
def worker(cur_policy, who_is_training, p1_index, p2_index, wining_rate, mutex,
           obses, rewards,
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

    # load player1 and 2 's model
    if who_is_training == 1:

        for i in range(p1_index + 1):
            basepolicy = PPOPolicy(is_training=False, k=K, model_path='model/base_policy/1/{0}/latest.ckpt'.format(i))
            policy_set1.append(basepolicy)

        base_policy2 = PPOPolicy(is_training=False, k=K, model_path='model/base_policy/2/{0}/latest.ckpt'.format(p2_index))
        mp1 = MixPolicy(max_policy=MAX_POLICY, is_training=False, k=K, model_path='model/weight1/latest.ckpt')

    else:

        for i in range(p2_index + 1):
            basepolicy = PPOPolicy(is_training=False, k=K, model_path='model/base_policy/2/{0}/latest.ckpt'.format(i))
            policy_set2.append(basepolicy)
        base_policy1 = PPOPolicy(is_training=False, k=K, model_path='model/base_policy/1/{0}/latest.ckpt'.format(p1_index))
        mp2 = MixPolicy(max_policy=MAX_POLICY, is_training=False, k=K, model_path='model/weight2/latest.ckpt')

    epoch = 0
    win_num = 0
    lose_num = 0
    p1_state = deque(maxlen=K)
    p2_state = deque(maxlen=K)

    while epoch < _SAMPLE_EPOCH:

        s, space = env.reset()
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

        traj_bp_actions = []
        traj_bp_v = []

        traj_mp_policy = []
        traj_mp_v = []

        # initial state
        for i in range(K):
            zero_state = np.zeros([41])
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

            if who_is_training == 1:
                p1, mp_v1 = mp1.get_policy(state1, p1_index)

                a1, mu_prob, bp_v1 = policy_set1[p1].get_action_value(state1)
                a2,_, bp_v2 = base_policy2.get_action_value(state2)

                pi_prob = policy_set1[cur_policy].get_act_prob(state1)[a1]

            else:
                p2, mp_v2 = mp2.get_policy(state2, p2_index)

                a1,_, bp_v1 = base_policy1.get_action_value(state1)
                a2, mu_prob, bp_v2 = policy_set2[p2].get_action_value(state2)

                pi_prob = policy_set2[cur_policy].get_act_prob(state2)[a2]

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            traj_r1+=r1
            traj_r2+=r2

            t += 1

            if t > 100:

                if who_is_training == 1:
                    if traj_r1 > traj_r2:
                        win_num += 1
                    elif traj_r2 > traj_r1:
                        lose_num += 1

                else:
                    if traj_r2 > traj_r1:
                        win_num += 1
                    elif traj_r1 > traj_r2:
                        lose_num += 1

                if who_is_training == 1:
                    traj_obs.append(state1)
                    traj_reward.append(r1)
                    traj_pi_prob.append(round(pi_prob,4))
                    traj_mu_prob.append(round(mu_prob,4))

                    traj_bp_actions.append(a1)
                    traj_bp_v.append(bp_v1)

                    traj_mp_policy.append(p1)
                    traj_mp_v.append(mp_v1)

                    p1, mp_v = mp1.get_policy(np.hstack((state1[41:], s1_)), p1_index)
                    _,_, bp_v = policy_set1[p1].get_action_value(np.hstack((state1[41:], s1_)))

                else:
                    traj_obs.append(state2)
                    traj_reward.append(r2)
                    traj_pi_prob.append(round(float(pi_prob),4))
                    traj_mu_prob.append(round(float(mu_prob),4))

                    traj_bp_actions.append(a2)
                    traj_bp_v.append(bp_v2)

                    traj_mp_policy.append(p2)
                    traj_mp_v.append(mp_v2)

                    p2, mp_v = mp2.get_policy(np.hstack((state2[41:], s2_)), p2_index)
                    _,_, bp_v = policy_set2[p2].get_action_value(np.hstack((state2[41:], s2_)))

                mutex.acquire()
                obses.append(traj_obs)
                bp_actions.append(traj_bp_actions)
                mp_policys.append(traj_mp_policy)
                rewards.append(traj_reward)

                traj_bp_v_next = traj_bp_v[1:] + [bp_v]
                traj_mp_v_next = traj_mp_v[1:] + [mp_v]

                bp_v_preds_next.append(traj_bp_v_next)
                mp_v_preds_next.append(traj_mp_v_next)

                traj_bp_gaes = bp_get_gaes(traj_reward, traj_bp_v, traj_bp_v_next, traj_pi_prob, traj_mu_prob)
                traj_bp_gaes = np.array(traj_bp_gaes).astype(dtype=np.float32)
                traj_bp_gaes = (traj_bp_gaes - traj_bp_gaes.mean()) / traj_bp_gaes.std()
                traj_bp_gaes = np.squeeze(traj_bp_gaes).tolist()
                bp_gaes.append(traj_bp_gaes)

                traj_mp_gaes = mp_get_gaes(traj_reward, traj_mp_v, traj_mp_v_next)
                traj_mp_gaes = np.array(traj_mp_gaes).astype(dtype=np.float32)
                traj_mp_gaes = (traj_mp_gaes - traj_mp_gaes.mean()) / traj_mp_gaes.std()
                traj_mp_gaes = np.squeeze(traj_mp_gaes).tolist()
                mp_gaes.append(traj_mp_gaes)

                mutex.release()
                epoch += 1
                break

            elif done:

                if t > 1:
                    if who_is_training == 1:
                        if traj_r1 > traj_r2:
                            win_num += 1
                        elif traj_r2 > traj_r1:
                            lose_num += 1
                    else:
                        if traj_r2 > traj_r1:
                            win_num += 1
                        elif traj_r1 > traj_r2:
                            lose_num += 1

                    if who_is_training == 1:
                        traj_obs.append(state1)
                        traj_reward.append(r1)
                        traj_pi_prob.append(pi_prob)
                        traj_mu_prob.append(mu_prob)

                        traj_bp_actions.append(a1)
                        traj_bp_v.append(bp_v1)

                        traj_mp_policy.append(p1)
                        traj_mp_v.append(mp_v1)

                        p1, mp_v = mp1.get_policy(np.hstack((state1[41:], s1_)), p1_index)
                        _,_, bp_v = policy_set1[p1].get_action_value(np.hstack((state1[41:], s1_)))

                    else:
                        traj_obs.append(state2)
                        traj_reward.append(r2)
                        traj_pi_prob.append(pi_prob)
                        traj_mu_prob.append(mu_prob)

                        traj_bp_actions.append(a2)
                        traj_bp_v.append(bp_v2)

                        traj_mp_policy.append(p2)
                        traj_mp_v.append(mp_v2)

                        p2, mp_v = mp2.get_policy(np.hstack((state2[41:], s2_)), p2_index)
                        _,_, bp_v = policy_set2[p2].get_action_value(np.hstack((state2[41:], s2_)))

                    mutex.acquire()
                    obses.append(traj_obs)
                    bp_actions.append(traj_bp_actions)
                    mp_policys.append(traj_mp_policy)
                    rewards.append(traj_reward)

                    traj_bp_v_next = traj_bp_v[1:] + [bp_v]
                    traj_mp_v_next = traj_mp_v[1:] + [mp_v]

                    bp_v_preds_next.append(traj_bp_v_next)
                    mp_v_preds_next.append(traj_mp_v_next)

                    traj_bp_gaes = bp_get_gaes(traj_reward, traj_bp_v, traj_bp_v_next, traj_pi_prob, traj_mu_prob)
                    traj_bp_gaes = np.array(traj_bp_gaes).astype(dtype=np.float32)
                    traj_bp_gaes = (traj_bp_gaes - traj_bp_gaes.mean()) / traj_bp_gaes.std()
                    traj_bp_gaes = np.squeeze(traj_bp_gaes).tolist()
                    bp_gaes.append(traj_bp_gaes)

                    traj_mp_gaes = mp_get_gaes(traj_reward, traj_mp_v, traj_mp_v_next)
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

                if who_is_training == 1:
                    traj_obs.append(state1)
                    traj_reward.append(r1)
                    traj_pi_prob.append(pi_prob)
                    traj_mu_prob.append(mu_prob)

                    traj_bp_actions.append(a1)
                    traj_bp_v.append(bp_v1)

                    traj_mp_policy.append(p1)
                    traj_mp_v.append(mp_v1)

                else:
                    traj_obs.append(state2)
                    traj_reward.append(r2)
                    traj_pi_prob.append(pi_prob)
                    traj_mu_prob.append(mu_prob)

                    traj_bp_actions.append(a2)
                    traj_bp_v.append(bp_v2)

                    traj_mp_policy.append(p2)
                    traj_mp_v.append(mp_v2)

            s1 = s1_
            s2 = s2_

    wining_rate.append((win_num-lose_num)/_SAMPLE_EPOCH)

def training_process(mp1_steps, mp2_steps, cur_policy_n, who_is_training):

    # first iteration
    manager = Manager()

    if cur_policy_n == 0:

        epoch = 0
        end_flag = False

        bp_steps = 0

        while epoch <= MAX_EPOCH:

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

            process_list = []

            # sample processes
            for i in range(_PROCESS_NUM):
                p = Process(target=worker, args=(cur_policy_n,
                                                 who_is_training,
                                                 0,
                                                 0,
                                                 wining_rate,
                                                 mutex,
                                                 obses,
                                                 rewards,
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

            if wr > _FIRST_END_THREHOLD:
                if end_flag:
                    break
                else:
                    end_flag = True
            else:
                end_flag = False

            bp_steps = optimize_by_childprocess(bp_steps, 0, cur_policy_n,who_is_training,obses,bp_actions,bp_gaes,rewards,bp_v_preds_next,mp_policys,mp_gaes,mp_v_preds_next)

        return 0

    # after first iteration
    else:
        #player1 is training
        if who_is_training == 1:

            bp_steps = 0
            mp_steps = mp1_steps

            # player1 trains a new base policy to defeat every policy of player2
            # to defeat n player2's base policy
            for policy2 in range(cur_policy_n):

                epoch = 0
                end_flag = False

                # to defeat one of opponent's bp
                while epoch <= MAX_EPOCH:

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

                    process_list = []
                    # sampling processes
                    for i in range(_PROCESS_NUM):
                        p = Process(target=worker, args=(cur_policy_n,
                                                         who_is_training,
                                                         cur_policy_n,
                                                         policy2,
                                                         wining_rate,
                                                         mutex,
                                                         obses,
                                                         rewards,
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
                    print('player1\'s bp{0} vs player2\'s bp{1}:{2} advantage, player1 is training!'.format(cur_policy_n, policy2, wr))

                    if wr > _END_THREHOLD:
                        if end_flag:
                            break
                        else:
                            end_flag = True
                    else:
                        end_flag = False

                    bp_steps, mp_steps = optimize_by_childprocess(bp_steps, mp_steps, cur_policy_n, who_is_training,obses,bp_actions,bp_gaes,rewards,
                                             bp_v_preds_next,mp_policys,mp_gaes,mp_v_preds_next)



        # player2 is training
        else:

            bp_steps = 0
            mp_steps = mp2_steps

            # to defeat n+1 player1's base policy
            for policy1 in range(cur_policy_n+1):

                epoch = 0
                end_flag = False

                while epoch <= MAX_EPOCH:

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

                    process_list = []

                    # sampling processes
                    for i in range(_PROCESS_NUM):
                        p = Process(target=worker, args=(cur_policy_n,
                                                         who_is_training,
                                                         policy1,
                                                         cur_policy_n,
                                                         wining_rate,
                                                         mutex,
                                                         obses,
                                                         rewards,
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
                    print('player1\'s bp{0} vs player2\'s bp{1}:{2} advantage, player2 is training!'.format(policy1, cur_policy_n, wr))

                    if wr > _END_THREHOLD:
                        if end_flag:
                            break
                        else:
                            end_flag = True
                    else:
                        end_flag = False

                    bp_steps, mp_steps = optimize_by_childprocess(bp_steps, mp_steps, cur_policy_n, who_is_training, obses, bp_actions, bp_gaes, rewards,
                                             bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next)

        return mp_steps

if __name__ == '__main__':
    flag = _NEED_MODEL

    if flag:
        #initialize weight
        mix_policy1 = MixPolicy(max_policy=MAX_POLICY, log_path='log/weight1/', k=K, model_path='model/weight1/latest.ckpt')
        mix_policy2 = MixPolicy(max_policy=MAX_POLICY, log_path='log/weight2/', k=K, model_path='model/weight2/latest.ckpt')

        #initialize base policy
        policy_set1 = []
        policy_set2 = []

        for i in range(MAX_POLICY):
           policy_set1.append(
               PPOPolicy(k=K,model_path='model/base_policy/1/{0}/latest.ckpt'.format(i), log_path='log/log1/{0}/'.format(i))
           )
           policy_set2.append(
               PPOPolicy(k=K,model_path='model/base_policy/2/{0}/latest.ckpt'.format(i), log_path='log/log2/{0}/'.format(i))
           )

        for i in range(MAX_POLICY):
           policy_set1[i].save_model()
           policy_set2[i].save_model()

        mix_policy1.save_model()
        mix_policy2.save_model()

    # training times for weight1 and 2
    mp1_n_training = 0
    mp2_n_training = 0

    for cur_policy_n in range(0,MAX_POLICY):

        print('---------iteration {0}  starts-------------'.format(cur_policy_n))
        mp1_n_training = training_process(mp1_n_training, mp2_n_training, cur_policy_n, who_is_training=1)

        print('player1 ends training base policy'.format(cur_policy_n))

        mp2_n_training = training_process(mp1_n_training, mp2_n_training, cur_policy_n, who_is_training=2)

        print('---------iteration {0}  ends---------------'.format(cur_policy_n))
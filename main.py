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
import tensorflow as tf

# params for mix policy training
K = 4        # combine history
MAX_POLICY = 20
MAX_EPOCH = 100
_END_THREHOLD = 0.55
_INDEX_THREHOLD = 0.5
_GAMMA = 0.99
_LAMBDA = 0.95

def optimize_worker(cur_policy_n, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next):

    if who_is_training == 1:
        base_policy1 = PPOPolicy(is_continuing=True, k=4,
                                 model_path='model/base_policy/1/{0}/latest.ckpt'.format(cur_policy_n))
        base_policy1.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy1.save_model()

        mp1 = MixPolicy(max_policy=MAX_POLICY, is_continuing=True, k=4, model_path='model/mix/weight1/latest.ckpt')
        mp1.train(obses, mp_policys, mp_gaes,rewards,mp_v_preds_next, cur_policy_n)
        mp1.save_model()
    else:
        base_policy2 = PPOPolicy(is_continuing=True, k=4,
                                 model_path='model/base_policy/2/{0}/latest.ckpt'.format(cur_policy_n))
        base_policy2.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy2.save_model()

        mp2 = MixPolicy(max_policy=MAX_POLICY, is_continuing=True, k=4, model_path='model/mix/weight2/latest.ckpt')
        mp2.train(obses, mp_policys, mp_gaes,rewards,mp_v_preds_next, cur_policy_n)
        mp2.save_model()

def optimize_worker_0(who_is_training, obses, bp_actions, bp_gaes, rewards, bp_v_preds_next):

    if who_is_training == 1:
        base_policy1 = PPOPolicy(is_continuing=True, k=4,
                                 model_path='model/base_policy/1/0/latest.ckpt')
        base_policy1.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy1.save_model()
    else:
        base_policy2 = PPOPolicy(is_continuing=True, k=4,
                                 model_path='model/base_policy/2/0/latest.ckpt')
        base_policy2.train(obses, bp_actions, bp_gaes, rewards, bp_v_preds_next)
        base_policy2.save_model()

def optimize_by_childprocess(cur_policy_n, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next):

    if cur_policy_n == 0:
        p = Process(target=optimize_worker_0,
                    args=(who_is_training, obses, bp_actions, bp_gaes, rewards, bp_v_preds_next,))
        p.start()
        p.join()

    else:

        p = Process(target=optimize_worker,
                    args=(cur_policy_n, who_is_training, obses, bp_actions, bp_gaes,
                    rewards, bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next,))
        p.start()
        p.join()


def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + _GAMMA * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + _GAMMA * _LAMBDA * gaes[t + 1]
    return gaes

# a worker to get samples
def worker(who_is_training, p1_index, p2_index, wining_rate, mutex,
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
            policy_set1.append(
                PPOPolicy(is_training=False, k=4, model_path='model/base_policy/1/{0}/latest.ckpt'.format(i))
            )
        base_policy2 = PPOPolicy(is_training=False,k=4, model_path='model/base_policy/2/{0}/latest.ckpt'.format(p2_index))
    else:

        for i in range(p2_index + 1):
            policy_set2.append(
                PPOPolicy(k=4, model_path='model/base_policy/2/{0}.latest.ckpt'.format(i), is_training=False)
            )
        base_policy1 = PPOPolicy(is_training=False,k=4, model_path='model/base_policy/1/{0}/latest.ckpt'.format(p1_index))


    mp1 = MixPolicy(max_policy= MAX_POLICY, is_training=False, k=4, model_path='model/mix/weight1/latest.ckpt')
    mp2 = MixPolicy(max_policy= MAX_POLICY, is_training=False, k=4, model_path='model/mix/weight2/latest.ckpt')

    epoch = 0
    win_num = 0
    p1_state = deque(maxlen=4)
    p2_state = deque(maxlen=4)

    while epoch < 200:

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

        traj_bp_actions = []
        traj_bp_v = []

        traj_mp_policy = []
        traj_mp_v = []

        # initial state
        for i in range(4):
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

                a1, bp_v1 = policy_set1[p1].get_action_value(state1)
                a2, bp_v2 = base_policy2.get_action_value(state2)
            else:
                p2, mp_v2 = mp2.get_policy(state2, p2_index)

                a1, bp_v1 = base_policy1.get_action_value(state1)
                a2, bp_v2 = policy_set2[p2].get_action_value(state2)

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            traj_r1+=r1
            traj_r2+=r2

            t += 1

            if t > 100:

                if who_is_training == 1:
                    if traj_r1 > traj_r2:
                        win_num += 1
                else:
                    if traj_r2 > traj_r1:
                        win_num ++ 1

                mutex.acquire()

                if who_is_training == 1:
                    traj_obs.append(state1)
                    traj_reward.append(r1)

                    traj_bp_actions.append(a1)
                    traj_bp_v.append(bp_v1)

                    traj_mp_policy.append(p1)
                    traj_mp_v.append(mp_v1)

                    p1, mp_v = mp1.get_policy(np.hstack((state1[41:], s1_)), p1_index)
                    _, bp_v = policy_set1[p1].get_action_value(np.hstack((state1[41:], s1_)))

                else:
                    traj_obs.append(state2)
                    traj_reward.append(r2)

                    traj_bp_actions.append(a2)
                    traj_bp_v.append(bp_v2)

                    traj_mp_policy.append(p2)
                    traj_mp_v.append(mp_v2)

                    p2, mp_v = mp2.get_policy(np.hstack((state2[41:], s2_)), p2_index)
                    _, bp_v = policy_set2[p2].get_action_value(np.hstack((state2[41:], s2_)))

                obses.append(traj_obs)
                bp_actions.append(traj_bp_actions)
                mp_policys.append(traj_mp_policy)
                rewards.append(traj_reward)

                traj_bp_v_next = traj_bp_v[1:] + [bp_v]
                traj_mp_v_next = traj_mp_v[1:] + [mp_v]

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
                    mutex.acquire()

                    if who_is_training == 1:
                        traj_obs.append(state1)
                        traj_reward.append(r1)

                        traj_bp_actions.append(a1)
                        traj_bp_v.append(bp_v1)

                        traj_mp_policy.append(p1)
                        traj_mp_v.append(mp_v1)

                        p1, mp_v = mp1.get_policy(np.hstack((state1[41:], s1_)), p1_index)
                        _, bp_v = policy_set1[p1].get_action_value(np.hstack((state1[41:], s1_)))

                    else:
                        traj_obs.append(state2)
                        traj_reward.append(r2)

                        traj_bp_actions.append(a2)
                        traj_bp_v.append(bp_v2)

                        traj_mp_policy.append(p2)
                        traj_mp_v.append(mp_v2)

                        p2, mp_v = mp2.get_policy(np.hstack((state2[41:], s2_)), p2_index)
                        _, bp_v = policy_set2[p2].get_action_value(np.hstack((state2[41:], s2_)))

                    obses.append(traj_obs)
                    bp_actions.append(traj_bp_actions)
                    mp_policys.append(traj_mp_policy)
                    rewards.append(traj_reward)

                    traj_bp_v_next = traj_bp_v[1:] + [bp_v]
                    traj_mp_v_next = traj_mp_v[1:] + [mp_v]

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

                if who_is_training == 1:
                    traj_obs.append(state1)
                    traj_reward.append(r1)

                    traj_bp_actions.append(a1)
                    traj_bp_v.append(bp_v1)

                    traj_mp_policy.append(p1)
                    traj_mp_v.append(mp_v1)

                else:
                    traj_obs.append(state2)
                    traj_reward.append(r2)

                    traj_bp_actions.append(a2)
                    traj_bp_v.append(bp_v2)

                    traj_mp_policy.append(p2)
                    traj_mp_v.append(mp_v2)

            s1 = s1_
            s2 = s2_

    wining_rate.append(win_num/250)

def training_process(cur_policy_n, who_is_training):

    # first iteration
    manager = Manager()

    if cur_policy_n == 0:

        epoch = 0
        index_epoch = 0
        index_flag = False
        wr = 0

        while epoch <= MAX_EPOCH and (epoch - index_epoch) <= 20:

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

            if wr >= _INDEX_THREHOLD:
                index_flag = True
                index_epoch = epoch

            if not index_flag:
                index_epoch = epoch

            tf.reset_default_graph()
            process_list = []

            # 4 sample processes
            for i in range(4):
                p = Process(target=worker, args=(who_is_training,
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

            wr = sum(wining_rate)/4

            optimize_by_childprocess(cur_policy_n,who_is_training,obses,bp_actions,bp_gaes,rewards,bp_v_preds_next,mp_policys,mp_gaes,mp_v_preds_next)

    # after first iteration
    else:
        #player1 is training
        if who_is_training == 1:

            # player1 trains a new base policy to defeat every policy of player2
            # to defeat n player2's base policy
            for policy2 in range(cur_policy_n):

                epoch = 0
                index_epoch = 0
                index_flag = False
                wr = 0

                while epoch <= MAX_EPOCH and (epoch - index_epoch) <= 20:

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

                    if wr >= _INDEX_THREHOLD:
                        index_flag = True
                        index_epoch = epoch

                    if index_flag == False:
                        index_epoch = epoch


                    process_list = []

                    # 4 sample processes
                    for i in range(4):
                        p = Process(target=worker, args=(who_is_training,
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

                    wr = sum(wining_rate) / 4

                    optimize_by_childprocess(cur_policy_n, who_is_training,obses,bp_actions,bp_gaes,rewards,
                                             bp_v_preds_next,mp_policys,mp_gaes,mp_v_preds_next)

        # player2 is training
        else:

            # to defeat n+1 player1's base policy
            for policy1 in range(cur_policy_n+1):
                epoch = 0
                index_epoch = 0
                index_flag = False
                wr = 0

                while epoch <= MAX_EPOCH and (epoch - index_epoch) <= 20:

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

                    if wr >= _INDEX_THREHOLD:
                        index_flag = True
                        index_epoch = epoch

                    if index_flag == False:
                        index_epoch = epoch

                    process_list = []

                    # 4 sample processes
                    for i in range(4):
                        p = Process(target=worker, args=(who_is_training,
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

                    wr = sum(wining_rate) / 4

                    optimize_by_childprocess(cur_policy_n, who_is_training, obses, bp_actions, bp_gaes, rewards,
                                             bp_v_preds_next, mp_policys, mp_gaes, mp_v_preds_next)

if __name__ == '__main__':


    # # initialize weight
    # mix_policy1 = MixPolicy(max_policy=MAX_POLICY, log_path='model/logs1/', k=K, model_path='model/mix/weight1/latest.ckpt')
    # mix_policy2 = MixPolicy(max_policy=MAX_POLICY, log_path='model/logs2/', k=K, model_path='model/mix/weight2/latest.ckpt')
    #
    # # initialize base policy
    # policy_set1 = []
    # policy_set2 = []
    #
    # for i in range(MAX_POLICY):
    #     policy_set1.append(
    #         PPOPolicy(k=K,model_path='model/base_policy/1/{0}/latest.ckpt'.format(i), log_path='model/base1_log/log{0}/'.format(i))
    #     )
    #     policy_set2.append(
    #         PPOPolicy(k=K,model_path='model/base_policy/2/{0}/latest.ckpt'.format(i), log_path='model/base2_log/log{0}/'.format(i))
    #     )
    #
    # for i in range(MAX_POLICY):
    #     policy_set1[i].save_model()
    #     policy_set2[i].save_model()
    #
    # mix_policy1.save_model()
    # mix_policy2.save_model()

    for cur_policy_n in range(0,MAX_POLICY):

        print('---------iteration {0}  starts-------------'.format(cur_policy_n))
        training_process(cur_policy_n, who_is_training=1)

        print('player1 ends training base policy'.format(cur_policy_n))

        training_process(cur_policy_n, who_is_training=2)

        print('---------iteration {0}  ends-------------'.format(cur_policy_n))

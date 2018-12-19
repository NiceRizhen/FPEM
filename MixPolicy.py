'''
  We try to mix policy with entropy weight method.

  @Author: Jingcheng Pang
'''



import numpy as np
from GameEnv import Game
from PPOPolicy import PPOPolicy
from multiprocessing import Process, Pool
from collections import deque

def evaluate_Mixpolicy(ps1, w1, ps2, w2, re1, re2):

    env = Game(8,8)
    n_p1 = len(ps1)
    n_p2 = len(ps2)
    reward1 = [0] * n_p1
    reward2 = [0] * n_p2

    select_n1 = [0] * n_p1
    select_n2 = [0] * n_p2

    epoch = 0
    while epoch < 1000:
        t = 0
        episode1 = []
        episode2 = []
        epoch += 1

        s, _ = env.reset()
        s1 = s[0]
        s2 = s[1]

        index1 = np.random.choice(n_p1, 1, p=w1)[0]
        index2 = np.random.choice(n_p2, 1, p=w2)[0]

        p1 = ps1[index1]
        p2 = ps2[index2]

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

            a1 = p1.choose_action_full_state(state1)
            a2 = p2.choose_action_full_state(state2)

            s1_, s2_, r1, r2, done = env.step(a1, a2)

            episode1.append(r1)
            episode2.append(r2)
            s1 = s1_
            s2 = s2_
            t += 1

            if t > 200:
                epoch -= 1
                break

            if done:
                break

        p1_state.clear()
        p2_state.clear()

        if done:
            reward1[index1] += sum(episode1)
            select_n1[index1] += 1

            reward2[index2] += sum(episode2)
            select_n2[index2] += 1

    for i in range(n_p1):
        reward1[i] = reward1[i]/select_n1[i] if select_n1[i] != 0 else 0

    for i in range(n_p2):
        reward2[i] = reward2[i]/select_n2[i] if select_n2[i] != 0 else 0

    re1.append(reward1)
    re2.append(reward2)

def compute_entropy(data0, _w):
    # sample numbers and policy numbers
    n,m=np.shape(data0)

    # normalization
    maxium=np.max(data0,axis=0)
    minium=np.min(data0,axis=0)
    data= (data0-minium)*1.0/(maxium-minium)

    sumzb=np.sum(data,axis=0)
    data=data/sumzb

    a=data*1.0
    a[np.where(data==0)]=0.0001

    e=(-1.0/np.log(n))*np.sum(data*np.log(a),axis=0)

    # update weight
    w=((1-e)*_w)/np.sum((1-e)*_w)

    return w

if __name__ == "__main__":

    # init policy set
    policy_set1 = []
    policy_set2 = []

    for i in range(5):
        policy_set1.append(
            PPOPolicy(is_training=False, k=4, model_path='model/1-3(150000)/{0}.ckpt'.format(i + 1))
        )

        policy_set2.append(
            PPOPolicy(is_training=False, k=4, model_path='model/2-3(150000)/{0}.ckpt'.format(i + 1))
        )

    # init weight1, weight2
    p1_n = len(policy_set1)
    p2_n = len(policy_set2)

    w1 = [1] * p1_n
    w1 = [w/sum(w1) for w in w1]
    w2 = [1] * p2_n
    w2 = [w/sum(w2) for w in w2]

    # init sample numbers
    K = 10

    iterations = 0
    while True:

        iterations += 1

        # init evaluate reward
        reward1 = []
        reward2 = []

        # process_list = []
        #
        # for i in range(K):
        #     process_list.append(
        #         Process(target=evaluate_Mixpolicy, args=(policy_set1,w1, policy_set2,w2, reward1, reward2,))
        #     )
        #
        # for p in process_list:
        #     p.start()
        #
        # for p in process_list:
        #     p.join()

        p = Pool(processes=3)
        for i in range(K):
            p.apply_async(evaluate_Mixpolicy(policy_set1,w1, policy_set2,w2, reward1, reward2))

        p.close()
        p.join()

        data1 = np.array(reward1)
        data2 = np.array(reward2)

        # optimize weight1
        w1 = compute_entropy(data1, w1)

        # optimize weight2
        w2 = compute_entropy(data2, w2)

        print('iteration(s):{0}'.format(iterations))
        print('weight1 : ', w1)
        print('weight2 : ', w2)

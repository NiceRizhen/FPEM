
import os
import matplotlib.pyplot as plt
from ExpertPolicy import ExpertPolicy
from GameEnv import Game
from PPOPolicy import PPOPolicy
from RandomPolicy import RandomPolicy


def run_with_policy(f_pi, g_pi):

    t = 0
    epoch = 0
    reward = []
    episode = []

    state = env.reset()
    while epoch < 100:

        a1, a2 = f_pi.choose_action(state)
        a3, a4 = g_pi.choose_action(state)
        state_, r1, r2 = env.step(a1, a2)
        episode.append(r2)
        state = state_
        t += 1
        if t % 300 == 0:
            reward.append(sum(episode))
            episode = []
            epoch += 1
            t = 0

    x = range(len(reward))
    plt.plot(x, reward, label="gvsfe")
    plt.xlabel('Episode')
    plt.ylabel('Return( vs fe)')
    plt.legend()
    plt.savefig('ge performance.jpg')

def alternating_training(training_pi, vs_pi, who_is_training:str, name:str):

    print('{0} is training.'.format(name))

    epoch = 0
    # EPOCH = 10000 if turn == 1 else 20000

    value_memory      = []
    state_memory      = []
    reward_memory     = []
    action1_memory    = []
    action2_memory    = []

    t = 0
    s = env.reset()
    while epoch < 20000:

        action, v = training_pi.get_action_value(s)
        if who_is_training is 'f':

            a1 = action[0]
            a2 = action[1]

            a3,a4 = vs_pi.choose_action(s)
        else:

            a3 = action[0]
            a4 = action[1]

            a1,a2 = vs_pi.choose_action(s)

        s_, r1, r2 = env.step(a1,a2,a3,a4)

        r = r1 if who_is_training is 'f' else r2

        state_memory.append(s)
        reward_memory.append(r)
        value_memory.append(v)
        action1_memory.append(action[0])
        action2_memory.append(action[1])

        s = s_
        t += 1
        if t % 300 == 0:
            act, v = training_pi.get_action_value(s)
            value_next_memory = value_memory[1:] + [v]

            training_pi.train(state_memory, action1_memory, action2_memory, reward_memory, value_memory, value_next_memory)

            value_memory = []
            state_memory = []
            reward_memory = []
            action1_memory = []
            action2_memory = []

            t = 0
            epoch += 1

    if who_is_training is 'f':
        training_pi.save_model('model/policy_for_f/{0}.ckpt'.format(name))
    else:
        training_pi.save_model('model/policy_for_g/{0}.ckpt'.format(name))

    print('{0} has finished training.'.format(name))


if __name__ == '__main__':
    env = Game(5, 5)

    pi_fe = PPOPolicy(is_training=False, model_path='model/policy_for_f/fevse.ckpt')
    pi_g = PPOPolicy(log_path='model/policy_for_g/gvsfelogs')

    alternating_training(training_pi=pi_g, vs_pi=pi_fe, who_is_training='g',name='gevsfe')
    # pi_fe.sess.close()
    # pi_g.sess.close()
    run_with_policy(pi_fe, pi_g)

    #
    # alternating_training(training_pi=pi_gt, vs_pi=pi_r, who_is_training='g', name='g1vsr')
    # alternating_training(training_pi=pi_f, vs_pi=pi_gt, who_is_training='f', name='f1vsg1')
    # alternating_training(training_pi=pi_gt, vs_pi=pi_f, who_is_training='g', name='g2vsf1')

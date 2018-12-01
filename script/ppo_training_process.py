
import random
from GameEnv import Game
from PPOPolicy import PPOPolicy

if __name__ == '__main__':
    env = Game(5, 5)

    pi = PPOPolicy(8)

    value_memory      = []
    state_memory      = []
    reward_memory     = []
    action1_memory    = []
    action2_memory    = []
    value_next_memory = []

    t = 0
    biggest = -1000000
    s = env.reset()
    while True:
        act, v = pi.get_action_value(s)

        s_, r1, r2 = env.step(random.randint(0,4),random.randint(0,4), act[0], act[1])

        state_memory.append(s)
        reward_memory.append(r2)
        value_memory.append(v)
        action1_memory.append(act[0])
        action2_memory.append(act[1])

        s = s_
        t += 1
        if t % 300 == 0:
            act, v =  pi.get_action_value(s)
            value_next_memory = value_memory[1:] + [v]

            pi.train(state_memory, action1_memory, action2_memory, reward_memory, value_memory, value_next_memory)

            batch_reward = sum(reward_memory)
            if batch_reward > biggest:
                biggest = batch_reward
                pi.save_model('model/ppo_for_g/ranvsppo{0}.ckpt'.format(biggest))

            value_memory      = []
            state_memory      = []
            reward_memory     = []
            action1_memory    = []
            action2_memory    = []
            value_next_memory = []

            t = 0
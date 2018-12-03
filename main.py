'''
  Entrance to Our Algorithm

  Runtime Environment:
    System         : Ubuntu 16.04
    python-version : 3.6.7
    tensorflow-gpu : 1.12.0
    IDE            : PyCharm

'''


from RandomPolicy import RandomPolicy
from GameEnv import Game
from PPOPolicy import PPOPolicy

if __name__ == '__main__':
    env = Game(8,8)

    pi = RandomPolicy()
    pi1 = PPOPolicy(log_path='model/policy_for_g/logs/')
    while True:

        value_memory = []
        state_memory = []
        reward_memory = []
        action_memory = []

        s1, s2 = env.reset()

        t = 0
        done = False

        while True:
            a1 = pi.choose_action(s1)
            a2, v = pi1.get_action_value(s2)

            s1,s2,r1,r2,done = env.step(a1,a2)

            state_memory.append(s2)
            reward_memory.append(r2)
            value_memory.append(v)
            action_memory.append(a2)

            t = t+1
            if done:
                if len(action_memory) < 10:
                    break

                act, v = pi1.get_action_value(s2)
                value_next_memory = value_memory[1:] + [v]

                pi1.train(state_memory, action_memory, reward_memory, value_memory, value_next_memory)



                t = 0
                break


from GameEnv import Game
from PPOPolicy import PPOPolicy
from visiable import Maze
from collections import deque
import time
import numpy as np

if __name__ == '__main__':
    space = []

    ga = Game(8,8)

    pi1 = PPOPolicy(is_training=False, model_path='model/2-3(150000)/1.ckpt', k=4)
    pi2 = PPOPolicy(is_training=False, model_path='model/2-3(150000)/5.ckpt', k=4)

    while True:
        s, space = ga.reset()
        s1 = s[0]
        s2 = s[1]
        env = Maze(space)
        t = 0

        p1_state = deque(maxlen=4)
        p2_state = deque(maxlen=4)

        for i in range(4):
            zero_state = np.zeros([45])
            p1_state.append(zero_state)
            p2_state.append(zero_state)

        while True:
            env.render()

            p1_state.append(s1)
            p2_state.append(s2)

            state1 = np.array([])
            for obs in p1_state:
                state1 = np.hstack((state1, obs))

            state2 = np.array([])
            for obs in p2_state:
                state2 = np.hstack((state2, obs))

            a1 = pi1.choose_action_full_state(state1)
            a2 = pi2.choose_action_full_state(state2)
            env.step(a1, a2, ga.who_takes_it)

            s1_,s2_,r1,r2,done = ga.step(a1,a2)

            s1 = s1_
            s2 = s2_
            t += 1
            if t > 100:
                break

            if done:
                env.render()
                time.sleep(0.2)
                break

        env.destroy()

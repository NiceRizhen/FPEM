'''
  This files contains visualization functions
  for observing the performance of our policy.
'''

import time
import numpy as np
import tkinter as tk
from GameEnv import Game
from PPOPolicy import PPOPolicy
from RandomPolicy import RandomPolicy

# visualization params
UNIT   = 40
MAZE_H = 10
MAZE_W = 10

# units in env
GROUND   = 0
WALL     = 1
OPPONENT = 2
TREASURE = 3

class Maze(tk.Tk, object):
    def __init__(self, space):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Grid World')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.wall = []
        self._build_maze(space)

    def _build_maze(self, space):
        self.wall = []
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        player = 0
        for x in range(10):
            for y in range(10):

                # create walls
                if space[x, y] == WALL:
                    wall_center = origin + np.array([UNIT * y, UNIT * x])
                    self.wall.append(self.canvas.create_rectangle(
                        wall_center[0] - 15, wall_center[1] - 15,
                        wall_center[0] + 15, wall_center[1] + 15,
                        fill='black'))

                # create treasure
                elif space[x, y] == TREASURE:
                    oval_center = origin + np.array([UNIT * y, UNIT * x])
                    self.oval = self.canvas.create_oval(
                        oval_center[0] - 15, oval_center[1] - 15,
                        oval_center[0] + 15, oval_center[1] + 15,
                        fill='yellow')

                # create players
                elif space[x, y] == OPPONENT:
                    if player is 0:
                        player += 1
                        rect_center = origin + np.array([UNIT * y, UNIT * x])
                        self.player1 = self.canvas.create_rectangle(
                            rect_center[0] - 15, rect_center[1] - 15,
                            rect_center[0] + 15, rect_center[1] + 15,
                            fill='red')
                    else:
                        rect_center = origin + np.array([UNIT * y, UNIT * x])
                        self.player2 = self.canvas.create_rectangle(
                            rect_center[0] - 15, rect_center[1] - 15,
                            rect_center[0] + 15, rect_center[1] + 15,
                            fill='blue')

        # pack all
        self.canvas.pack()

    def step(self, action1, action2):
        s1 = self.canvas.coords(self.player1)
        s2 = self.canvas.coords(self.player2)

        # player1
        base_action1 = np.array([0, 0])
        if action1 == 0:   # up
            if s1[1] > UNIT:
                base_action1[1] -= UNIT
        elif action1 == 1:   # down
            if s1[1] < (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
        elif action1 == 2:   # left
            if s1[0] > UNIT:
                base_action1[0] -= UNIT
        elif action1 == 3:   # right
            if s1[0] < (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT

        # player2
        base_action2 = np.array([0, 0])
        if action2 == 0:   # up
            if s2[1] > UNIT:
                base_action2[1] -= UNIT
        elif action2 == 1:   # down
            if s2[1] < (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action2 == 2:   # left
            if s2[0] > UNIT:
                base_action2[0] -= UNIT
        elif action2 == 3:   # right
            if s2[0] < (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT

        self.canvas.move(self.player1, base_action1[0], base_action1[1])  # move player1
        self.canvas.move(self.player2, base_action2[0], base_action2[1])  # move player1

        s_1 = self.canvas.coords(self.player1)  # next state
        s_2 = self.canvas.coords(self.player2)  # next state

        done = False
        if s_1 == self.canvas.coords(self.oval):
            done = True
        elif s_1 in [self.canvas.coords(wal) for wal in self.wall]:
            self.canvas.move(self.player1, -base_action1[0], -base_action1[1])

        if s_2 == self.canvas.coords(self.oval):
            done = True
        elif s_2 in [self.canvas.coords(wal) for wal in self.wall]:
            self.canvas.move(self.player2, -base_action2[0], -base_action2[1])

        return done

    def render(self):
        time.sleep(0.1)
        self.update()

if __name__ == '__main__':
    space = []

    ga = Game(8,8)

    pi_random = RandomPolicy()
    pi_ppo = PPOPolicy(is_training=False, model_path='model/ppo-20000.ckpt')

    while True:
        s, space = ga.reset()
        s1 = s[0]
        s2 = s[1]
        env = Maze(space)
        t = 0
        while True:
            env.render()
            a1 = pi_random.choose_action(s1)
            a2 = pi_ppo.choose_action(s2)
            _ = env.step(a1, a2)

            s1_,s2_,r1,r2,done = ga.step(a1,a2)

            s1 = s1_
            s2 = s2_
            t += 1
            if t > 1500:
                break

            if done:
                break

        env.destroy()

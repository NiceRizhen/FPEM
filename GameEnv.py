'''
 This file is to implement the game environment.
 There are two players aim for finding 2 treasures in the grid world.
 You can visualize the env vid run code in visualization.py.

 @Author: pjc.

'''

import random
import numpy as np

# action space:
MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3

INIT_MAP = [
[[3, 4], [3, 5], [3, 6], [5, 4], [6, 3], [6, 4], [6, 7], [7, 7], [8, 7]] ,
[[3, 2], [3, 3], [4, 2], [5, 4], [5, 5], [6, 4], [7, 8], [8, 7], [8, 8]] ,
[[2, 5], [3, 5], [4, 5], [5, 3], [5, 4], [5, 5], [6, 7], [7, 6], [7, 7]] ,
[[1, 1], [2, 1], [2, 2], [3, 3], [4, 2], [4, 3], [6, 3], [7, 2], [7, 3]] ,
[[1, 7], [1, 8], [2, 7], [3, 5], [4, 5], [5, 5], [8, 2], [8, 3], [8, 4]] ,
[[2, 3], [2, 4], [2, 5], [2, 7], [3, 7], [3, 8], [4, 6], [5, 6], [6, 6]] ,
[[1, 3], [1, 4], [2, 3], [2, 4], [2, 5], [3, 4], [4, 7], [4, 8], [5, 7]] ,
[[2, 2], [2, 3], [3, 1], [3, 2], [4, 1], [5, 1], [6, 3], [6, 4], [7, 3]] ,
[[1, 5], [1, 6], [2, 5], [4, 5], [4, 6], [5, 6], [8, 6], [8, 7], [8, 8]] ,
[[2, 6], [3, 5], [3, 6], [5, 2], [5, 3], [5, 4], [5, 7], [6, 7], [6, 8]] ,
[[1, 7], [1, 8], [2, 1], [2, 7], [3, 1], [3, 2], [8, 5], [8, 6], [8, 7]] ,
[[2, 2], [3, 2], [3, 3], [3, 4], [3, 7], [3, 8], [4, 4], [4, 8], [5, 4]] ,
[[2, 5], [2, 6], [2, 7], [6, 2], [6, 3], [6, 4], [6, 7], [6, 8], [7, 8]] ,
[[3, 3], [3, 4], [3, 5], [5, 4], [5, 5], [5, 6], [7, 3], [7, 4], [8, 3]] ,
[[1, 4], [2, 4], [2, 5], [4, 2], [5, 2], [5, 3], [5, 4], [5, 5], [6, 4]] ,
[[1, 2], [1, 3], [1, 4], [5, 4], [6, 4], [7, 4], [7, 6], [8, 5], [8, 6]] ,
[[3, 4], [4, 4], [5, 4], [6, 4], [7, 1], [7, 4], [7, 5], [8, 1], [8, 2]] ,
[[3, 4], [3, 5], [3, 6], [3, 7], [4, 7], [4, 8], [6, 5], [7, 5], [7, 6]] ,
[[2, 2], [2, 8], [3, 2], [3, 3], [3, 4], [3, 7], [3, 8], [4, 3], [4, 4]] ,
[[2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [7, 6], [7, 7], [8, 6]]]

'''
  For unit above the ground, we set 0 for ground, 1 for wall, 
  2 for opponent, 3 for treasure, 4 for opponent with treasure.
'''
GROUND   = 0
WALL     = 1
OPPONENT = 2
TREASURE = 3
OANDTREA = 4

# wall's shape, 0 for line, 1 for curve
LINE  = 0
CURVE = 1

class Game():

    def __init__(self, xlen, ylen):

        # Game env list
        self.xlen = xlen
        self.ylen = ylen
        self.x_bound = self.xlen + 1
        self.y_bound = self.ylen + 1
        self.space = np.zeros([self.xlen + 2, self.ylen + 2], dtype=np.int32)

        self.x1, self.x2, self.y1, self.y2 = 1,1,1,1

        # init the env
        self.reset()

    def get_obs(self):

        sp = self.space
        x = self.x1
        y = self.y1

        obs1 = np.zeros([12,5])
        obs1[0, sp[x-1,y-1]] = 1
        obs1[1, sp[x-1,y]]   = 1
        obs1[2, sp[x-1,y+1]] = 1
        obs1[3, sp[x,y-1]]   = 1
        obs1[4, sp[x,y+1]]   = 1
        obs1[5, sp[x+1,y-1]] = 1
        obs1[6, sp[x+1,y]]   = 1
        obs1[7, sp[x+1,y+1]] = 1

        if x-2 >= 0:
            obs1[8, sp[x-2, y]] = 1
        if x+2 <= self.x_bound:
            obs1[9, sp[x+2, y]] = 1
        if y-2 >= 0:
            obs1[10, sp[x, y-2]] = 1
        if y+2 <= self.y_bound:
            obs1[11, sp[x, y+2]] = 1

        obs1 = obs1.flatten()

        x = self.x2
        y = self.y2

        obs2 = np.zeros([12,5])
        obs2[0, sp[x-1,y-1]] = 1
        obs2[1, sp[x-1,y]]   = 1
        obs2[2, sp[x-1,y+1]] = 1
        obs2[3, sp[x,y-1]]   = 1
        obs2[4, sp[x,y+1]]   = 1
        obs2[5, sp[x+1,y-1]] = 1
        obs2[6, sp[x+1,y]]   = 1
        obs2[7, sp[x+1,y+1]] = 1

        if x-2 >= 0:
            obs2[8, sp[x-2, y]] = 1
        if x+2 <= self.x_bound:
            obs2[9, sp[x+2, y]] = 1
        if y-2 >= 0:
            obs2[10, sp[x, y-2]] = 1
        if y+2 <= self.y_bound:
            obs2[11, sp[x, y+2]] = 1
        obs2 = obs2.flatten()

        if self.who_takes_it == 0:
            obs1 = np.hstack((obs1, [0]))
            obs2 = np.hstack((obs2, [0]))

        elif self.who_takes_it == 1:
            obs1 = np.hstack((obs1, [1]))
            obs2 = np.hstack((obs2, [0]))

        else:
            obs1 = np.hstack((obs1, [0]))
            obs2 = np.hstack((obs2, [1]))

        return obs1.flatten(), obs2.flatten()

    def move(self, x, y, action):

        if action == MOVE_UP:
            if self.space[x-1, y] == WALL:
                pass
            else:
                self.space[x,y] = GROUND
                x = x-1

        elif action == MOVE_DOWN:
            if self.space[x+1, y] == WALL:
                pass
            else:
                self.space[x, y] = GROUND
                x = x+1

        elif action == MOVE_LEFT:
            if self.space[x, y-1] == WALL:
                pass
            else:
                self.space[x,y] = GROUND
                y = y-1

        elif action == MOVE_RIGHT:
            if self.space[x, y+1] == WALL:
                pass
            else:
                self.space[x,y] = GROUND
                y = y+1
        else:
            print('wrong action')

        return x, y

    def step(self, a1, a2):

        r1 = 0

        x, y = self.move(self.x1, self.y1, a1)
        self.x1 = x
        self.y1 = y

        x, y = self.move(self.x2, self.y2, a2)
        self.x2 = x
        self.y2 = y

        done = False

        # two players come across
        if self.x1 == self.x2 and self.y1 == self.y2:

            # they both get the treasure
            if self.space[self.x1,self.y1] == TREASURE:
                self.treasure_n -= 1

            # a guy robs the other guy
            elif  self.treasure_n == 1:
                if self.who_takes_it == 1:
                    self.who_takes_it = 2
                    r1 = -2
                elif self.who_takes_it == 2:
                    self.who_takes_it = 1
                    r1 = 2

            # nothing happens
            else:
                pass

            self.space[self.x1, self.y1] = OPPONENT

        # they don't walk into the same grid
        else:

            # both p1 and p2 find treasure
            if self.space[self.x1, self.y1] == TREASURE and self.space[self.x2, self.y2] == TREASURE:

                self.treasure_n -= 2
                self.space[self.x1, self.y1] = OPPONENT
                self.space[self.x2, self.y2] = OPPONENT

            # only p1 finds treasure
            elif self.space[self.x1, self.y1] == TREASURE:
                if self.treasure_n == 1:
                    if self.who_takes_it == 1:
                        r1 = 2
                    else:
                        r1 += 1

                    self.treasure_n -= 1
                    self.space[self.x1, self.y1] = OPPONENT
                    self.space[self.x2, self.y2] = OPPONENT

                else:
                    r1 = 1
                    self.who_takes_it = 1
                    self.treasure_n -= 1

                    self.space[self.x1, self.y1] = OANDTREA
                    self.space[self.x2, self.y2] = OPPONENT

            # only p2 finds treasure
            elif self.space[self.x2, self.y2] == TREASURE:
                if self.treasure_n == 1:
                    if self.who_takes_it == 2:
                        r1 = -2
                    else:
                        r1 = -1

                    self.treasure_n -= 1
                    self.space[self.x1, self.y1] = OPPONENT
                    self.space[self.x2, self.y2] = OPPONENT

                else:
                    r1 = -1
                    self.who_takes_it = 2
                    self.treasure_n -= 1

                    self.space[self.x2, self.y2] = OANDTREA
                    self.space[self.x1, self.y1] = OPPONENT

            # nothing happens
            else:
                if self.who_takes_it == 1:
                    self.space[self.x1, self.y1] = OANDTREA
                    self.space[self.x2, self.y2] = OPPONENT

                elif self.who_takes_it == 2:
                    self.space[self.x2, self.y2] = OANDTREA
                    self.space[self.x1, self.y1] = OPPONENT

                else:
                    self.space[self.x1, self.y1] = OPPONENT
                    self.space[self.x2, self.y2] = OPPONENT

        if self.treasure_n == 0:
            done = True

        s1 , s2 = self.get_obs()
        r2 = -r1

        return s1,s2,r1,r2,done

    def reset(self):

        # treasures and who takes 1st treasure
        self.who_takes_it = 0
        self.treasure_n = 2

        # init edge wall
        self.space = np.zeros([self.xlen+2, self.ylen+2], dtype=np.int32)
        self.space[0, :] = WALL
        self.space[self.xlen+1, :] = WALL
        self.space[:, 0] = WALL
        self.space[:, self.ylen+1] = WALL

        map = INIT_MAP[random.randint(0,19)]
        for pos in map:
            self.space[pos[0], pos[1]] = WALL

        # init player1
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.space[x,y]==WALL:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.x1 = x
        self.y1 = y
        self.space[x,y] = OPPONENT

        # init player2
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.space[x,y]==WALL or self.space[x,y] == OPPONENT:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.x2 = x
        self.y2 = y
        self.space[x,y] = OPPONENT

        # init treasure1
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.space[x,y] == WALL or \
              self.space[x,y] == OPPONENT:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.space[x, y] = TREASURE

        # init treasure2
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.space[x,y] == WALL or \
              self.space[x,y] == OPPONENT or\
              self.space[x,y] == TREASURE:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.space[x, y] = TREASURE

        return self.get_obs(), self.space

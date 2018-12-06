# This file is to implement the game environment


import numpy as np
import random
import math

# action space:
MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3

'''
  For state, we set 0 for ground, 
  1 for wall, 2 for opponent, 3 for treasure
'''
GROUND   = 0
WALL     = 1
OPPONENT = 2
TREASURE = 3

# wall's shape, 0 for line, 1 for curve
LINE  = 0
CURVE = 1

class Game():

    def __init__(self, xlen, ylen):

        # Game env list
        self.xlen = xlen
        self.ylen = ylen
        self.space = np.zeros([self.xlen + 2, self.ylen + 2], dtype=np.int32)

        self.x1, self.x2, self.y1, self.y2 = 1,1,1,1

        # init the env
        self.reset()

    def get_obs(self):

        sp = self.space
        x = self.x1
        y = self.y1

        obs1 = np.zeros([8,4])
        obs1[0, sp[x-1,y-1]] = 1
        obs1[1, sp[x-1,y]]   = 1
        obs1[2, sp[x-1,y+1]] = 1
        obs1[3, sp[x,y-1]]   = 1
        obs1[4, sp[x,y+1]]   = 1
        obs1[5, sp[x+1,y-1]] = 1
        obs1[6, sp[x+1,y]]   = 1
        obs1[7, sp[x+1,y+1]] = 1

        x = self.x2
        y = self.y2
        obs2 = np.zeros([8,4])
        obs2[0, sp[x-1,y-1]] = 1
        obs2[1, sp[x-1,y]]   = 1
        obs2[2, sp[x-1,y+1]] = 1
        obs2[3, sp[x,y-1]]   = 1
        obs2[4, sp[x,y+1]]   = 1
        obs2[5, sp[x+1,y-1]] = 1
        obs2[6, sp[x+1,y]]   = 1
        obs2[7, sp[x+1,y+1]] = 1

        return obs1.flatten(), obs2.flatten()

    def move(self, x, y, action):

        done = False

        if action == MOVE_UP:
            if self.space[x-1, y] == WALL:
                pass
            elif self.space[x-1, y]==TREASURE:
                done = True
                self.space[x,y] = GROUND
                x = x-1
            else:
                self.space[x,y] = GROUND
                x = x-1

        elif action == MOVE_DOWN:
            if self.space[x+1, y] == WALL:
                pass
            elif self.space[x+1, y]==TREASURE:
                done = True
                self.space[x,y] = GROUND
                x = x+1
            else:
                self.space[x,y] = GROUND
                x = x+1

        elif action == MOVE_LEFT:
            if self.space[x, y-1] == WALL:
                pass
            elif self.space[x, y-1]==TREASURE:
                done = True
                self.space[x,y] = GROUND
                y = y-1
            else:
                self.space[x,y] = GROUND
                y = y-1

        elif action == MOVE_RIGHT:
            if self.space[x, y+1] == WALL:
                pass
            elif self.space[x, y+1]==TREASURE:
                done = True
                self.space[x,y] = GROUND
                y = y+1
            else:
                self.space[x,y] = GROUND
                y = y+1
        else:
            print('wrong action')

        return done, x, y

    def step(self, a1, a2):

        r1, r2 = 0, 0

        done1, x, y = self.move(self.x1, self.y1, a1)
        self.x1 = x
        self.y1 = y

        done2, x, y = self.move(self.x2, self.y2, a2)
        self.x2 = x
        self.y2 = y

        self.space[self.x1,self.y1] = OPPONENT
        self.space[self.x2,self.y2] = OPPONENT

        done = done1 or done2

        if done1:
            r1 = 1
            r2 = -1
        elif done2:
            r1 = -1
            r2 = 1

        if done1 and done2:
            r1 = 1
            r2 = 1

        s1 , s2 = self.get_obs()

        return s1,s2,r1,r2,done

    def wall_num(self, x, y):

        walls = 0

        walls = walls + 1 if self.space[x - 1, y] == WALL else walls
        walls = walls + 1 if self.space[x + 1, y] == WALL else walls
        walls = walls + 1 if self.space[x, y - 1] == WALL else walls
        walls = walls + 1 if self.space[x, y + 1] == WALL else walls

        return walls

    def reset(self):

        # init edge wall
        self.space = np.zeros([self.xlen+2, self.ylen+2], dtype=np.int32)
        self.space[0, :] = WALL
        self.space[self.xlen+1, :] = WALL
        self.space[:, 0] = WALL
        self.space[:, self.ylen+1] = WALL

        # init mid wall
        i = 0
        while i < 3:

            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

            # create line wall
            if random.randint(0,1) == LINE:

                form = random.randint(0,1)
                if form is 0:
                    while self.space[x,y] == WALL or self.space[x,y-1] == WALL or self.space[x,y+1] == WALL:
                        x = random.randint(1, self.xlen)
                        y = random.randint(1, self.ylen)

                    self.space[x, y-1] = WALL
                    self.space[x, y] = WALL
                    self.space[x, y+1] = WALL

                else:
                    while self.space[x-1,y] == WALL or self.space[x,y] == WALL or self.space[x+1,y] == WALL:
                        x = random.randint(1, self.xlen)
                        y = random.randint(1, self.ylen)

                    self.space[x-1, y] = WALL
                    self.space[x, y] = WALL
                    self.space[x+1, y] = WALL

            # create curve wall
            else:

                form = random.randint(0,3)

                if form == 0:
                    while self.space[x+1, y] == WALL or self.space[x, y] == WALL or self.space[x, y+1] == WALL:
                        x = random.randint(1, self.xlen)
                        y = random.randint(1, self.ylen)

                    self.space[x+1, y] = WALL
                    self.space[x, y] = WALL
                    self.space[x , y+1] = WALL

                elif form == 1:
                    while self.space[x+1, y] == WALL or self.space[x, y] == WALL or self.space[x, y-1] == WALL:
                        x = random.randint(1, self.xlen)
                        y = random.randint(1, self.ylen)

                    self.space[x+1, y] = WALL
                    self.space[x, y] = WALL
                    self.space[x, y-1] = WALL

                elif form == 2:
                    while self.space[x-1, y] == WALL or self.space[x, y] == WALL or self.space[x, y+1] == WALL:
                        x = random.randint(1, self.xlen)
                        y = random.randint(1, self.ylen)

                    self.space[x-1, y] = WALL
                    self.space[x, y] = WALL
                    self.space[x, y+1] = WALL

                else:
                    while self.space[x-1, y] == WALL or self.space[x, y] == WALL or self.space[x, y-1] == WALL:
                        x = random.randint(1, self.xlen)
                        y = random.randint(1, self.ylen)

                    self.space[x-1, y] = WALL
                    self.space[x, y] = WALL
                    self.space[x, y-1] = WALL

            i += 1

        # init player1
        x = random.randint(1, self.xlen-1)
        y = random.randint(1, self.ylen)

        while self.space[x,y]==WALL:
            x = random.randint(1, self.xlen-1)
            y = random.randint(1, self.ylen)

        self.x1 = x
        self.y1 = y
        self.space[x,y] = OPPONENT

        # init player2
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.space[x,y]==WALL or self.space[x,y] == OPPONENT or x <= self.x1:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.x2 = x
        self.y2 = y
        self.space[x,y] = OPPONENT

        # init treasure
        x = random.randint(3, self.xlen-2)
        y = random.randint(3, self.ylen-2)

        while self.space[x,y] == WALL or \
              self.space[x,y] == OPPONENT:
            x = random.randint(3, self.xlen-2)
            y = random.randint(3, self.ylen-2)

        self.space[x, y] = TREASURE

        return self.get_obs(), self.space
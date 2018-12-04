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
# units in env
GROUND   = 0
WALL     = 1
OPPONENT = 2
TREASURE = 3

class Game():

    def __init__(self, xlen, ylen):

        # Game env list
        self.xlen = xlen
        self.ylen = ylen
        self.space = np.zeros([self.xlen + 2, self.ylen + 2], dtype=np.int32)

        self.x1, self.x2, self.y1, self.y2 = 1,1,1,1

        self.walls = (int)(math.sqrt(xlen*ylen)/2)

        # init the env
        self.reset()

    def get_obs(self):

        sp = self.space
        x = self.x1
        y = self.y1

        obs1 = [sp[x-1,y-1], sp[x-1,y], sp[x-1,y+1], sp[x,y-1], sp[x,y+1], sp[x+1,y-1], sp[x+1,y], sp[x+1,y+1]]

        x = self.x2
        y = self.y2
        obs2 = [sp[x-1,y-1], sp[x-1,y], sp[x-1,y+1], sp[x,y-1], sp[x,y+1], sp[x+1,y-1], sp[x+1,y], sp[x+1,y+1]]

        return obs1, obs2

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
        for i in range(self.walls):
            x = random.randint(2, self.xlen-1)
            y = random.randint(2, self.ylen-1)
            self.space[x, y] = WALL

        # init player1
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.wall_num(x,y)==4 or self.space[x,y]==WALL:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.x1 = x
        self.y1 = y
        self.space[x,y] = OPPONENT

        # init player2
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.wall_num(x,y)==4 or self.space[x,y]==WALL:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.x2 = x
        self.y2 = y
        self.space[x,y] = OPPONENT

        # init treasure
        x = random.randint(1, self.xlen)
        y = random.randint(1, self.ylen)

        while self.space[x,y] == WALL or \
              self.space[x,y] == OPPONENT or \
              self.wall_num(x,y) == 4:
            x = random.randint(1, self.xlen)
            y = random.randint(1, self.ylen)

        self.space[x, y] = TREASURE

        return self.get_obs(), self.space
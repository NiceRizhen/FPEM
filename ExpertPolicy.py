'''
  Policy for Expert with simple logic from human experience.
  For hunter, we just let one g get closer to the nearest f
  while the rest g is chasing for the other prey.

'''

# action space:
MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3
MOVE_STAY  = 4

import random
from BasePolicy import policy

class ExpertPolicy(policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, state):

        # Disturbing noise with epsilon
        if random.random() > self.epsilon:
            x_f1, y_f1, x_f2, y_f2, x_g1, y_g1, x_g2, y_g2 = \
                state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],

            d_g1_f1 = abs(x_g1-x_f1) + abs(y_g1-y_f1)
            d_g1_f2 = abs(x_g1-x_f2) + abs(y_g1-y_f2)
            d_g2_f1 = abs(x_g2-x_f1) + abs(y_g2-y_f1)
            d_g2_f2 = abs(x_g2-x_f2) + abs(y_g2-y_f2)

            act_g1_f1 = self.chase(x_f1,y_f1,x_g1,y_g1)
            act_g1_f2 = self.chase(x_f2,y_f2,x_g1,y_g1)
            act_g2_f1 = self.chase(x_f1,y_f1,x_g2,y_g2)
            act_g2_f2 = self.chase(x_f2,y_f2,x_g2,y_g2)

            # if g1 and g2 are closer to different f
            if d_g1_f1 < d_g1_f2 and d_g2_f2 < d_g2_f1:
                action1 = act_g1_f1
                action2 = act_g2_f2
            elif d_g1_f2 < d_g1_f1 and d_g2_f1 < d_g2_f2:
                action1 = act_g1_f2
                action2 = act_g2_f1

            # g1 and g2 are closer to same f
            elif d_g1_f1 < d_g1_f2 and d_g2_f1 < d_g2_f2:
                if d_g1_f1 < d_g2_f1:
                    action1 = act_g1_f1
                    action2 = act_g2_f2
                else:
                    action1 = act_g1_f2
                    action2 = act_g2_f1
            else:
                if d_g1_f2 < d_g2_f2:
                    action1 = act_g1_f2
                    action2 = act_g2_f1
                else:
                    action1 = act_g1_f1
                    action2 = act_g2_f2
        else:
            action1 = random.randint(0,4)
            action2 = random.randint(0,4)

        return action1, action2

    def chase(self, x_tar, y_tar, x_self, y_self):

        if x_self==x_tar and y_self==y_tar:
            action = MOVE_STAY

        elif x_self==x_tar:
            action = MOVE_DOWN if y_tar>y_self else MOVE_UP

        elif y_self==y_tar:
            action = MOVE_RIGHT if x_tar>x_self else MOVE_LEFT

        else:
            act1 = MOVE_DOWN if y_tar>y_self else MOVE_UP
            act2 = MOVE_RIGHT if x_tar>x_self else MOVE_LEFT
            action = random.choice([act1,act2])

        return action

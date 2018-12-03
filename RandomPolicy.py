import random
from BasePolicy import policy

class RandomPolicy(policy):

    def choose_action(self, state):
        a = random.randint(0, 3)

        return a

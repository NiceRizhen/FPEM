import random
from BasePolicy import policy

class RandomPolicy(policy):

    def choose_action(self, state):
        a1 = random.randint(0, 4)
        a2 = random.randint(0, 4)

        return a1, a2

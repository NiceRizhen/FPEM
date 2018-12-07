'''
  We typically use this file to implement
  generating basic policy with Residual Method
'''

from PPOPolicy import PPOPolicy

if __name__ == '__main__':
    policy_set = []
    MAX = 5

    for iteration in range(MAX):
        np = PPOPolicy(k=4, )

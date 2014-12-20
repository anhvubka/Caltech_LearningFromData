__author__ = 'Le'
import random


class Coin(object):

    def __init__(self):
        total_head = 0
        for i in range(10):
            a = random.randint(0, 1)
            total_head += a
        self.pro = total_head/10.0


def toss_coins(a):
    n = random.randint(0, a - 1)
    p1 = 0
    p_min = 1
    p_rand = 0
    for i in range(a):
        coin = Coin()
        if i == 0:
            p1 = coin.pro
        if i == n:
            p_rand = coin.pro
        if coin.pro < p_min:
            p_min = coin.pro

    return p1, p_min, p_rand
N = 10000
p1_average = 0
p_min_average = 0
p_rand_average = 0
for i in range(N):
    result = toss_coins(1000)
    print result
    p1_average += result[0]
    p_min_average += result[1]
    p_rand_average += result[2]
print p1_average
print p_min_average
print p_rand_average








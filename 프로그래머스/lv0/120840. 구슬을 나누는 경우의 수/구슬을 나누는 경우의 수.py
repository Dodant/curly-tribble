from functools import reduce

def solution(balls, share):
    if balls == 1: return 1
    if share == 1: return reduce(lambda x,y: x*y, range(share, balls+1))
    return reduce(lambda x,y: x*y, range(balls-share+1, balls+1)) // reduce(lambda x,y: x*y, range(1, share+1))
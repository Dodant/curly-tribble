from functools import reduce

def solution(num_list):
    if len(num_list) >= 11:
        return sum(num_list)
    return reduce(lambda a, b: a*b, num_list)
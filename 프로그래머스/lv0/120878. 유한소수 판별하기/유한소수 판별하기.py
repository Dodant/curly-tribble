import math

def divisor(a):
    return [i for i in range(1, a+1) if a % i == 0]


def solution(a, b):
    if int(a/b) == a/b: return 1
    div = divisor(b // math.gcd(a,b))
    cdv = [i for i in div if len(divisor(i)) == 2]
    if cdv in [[2],[5],[2,5]]: return 1
    return 2
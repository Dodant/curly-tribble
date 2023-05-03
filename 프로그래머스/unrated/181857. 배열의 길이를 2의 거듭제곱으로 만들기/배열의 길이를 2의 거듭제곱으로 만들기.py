import math

def solution(arr):
    if int(math.log2(len(arr))) == math.log2(len(arr)):
        return arr
    arr += [0] * (2 ** int(math.log2(len(arr)) + 1) - len(arr))
    return arr
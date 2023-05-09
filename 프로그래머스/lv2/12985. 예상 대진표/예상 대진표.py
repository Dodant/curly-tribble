import math

def solution(n,a,b):
    answer = 0
    while True:
        a = math.ceil(a/2)
        b = math.ceil(b/2)
        if a == b: 
            answer += 1
            break
        answer += 1

    return answer
def lcm(a, b):
    return a*b//gcd(a,b)

def gcd(a, b):
    if b == 0: return a 
    return gcd(b, a%b)

def solution(arr):
    answer = 1
    for i in arr:
        answer = lcm(answer, i)
    return answer
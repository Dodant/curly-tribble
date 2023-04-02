def solution(price, money, count):
    return max(price*sum((i for i in range(1, count+1))) - money, 0)
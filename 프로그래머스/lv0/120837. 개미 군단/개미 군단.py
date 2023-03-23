def solution(hp):
    cnt = 0
    cnt += hp//5
    hp -= (hp//5)*5
    cnt += hp//3
    hp -= (hp//3)*3
    cnt += hp
    return cnt
def solution(ls):
    cnt = [0]*1000
    for i in ls: cnt[i] += 1
    if cnt.count(max(cnt)) > 1 :
        return -1
    else:
        return cnt.index(max(cnt))
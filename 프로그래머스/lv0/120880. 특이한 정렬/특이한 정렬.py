def solution(numlist, n):
    ls = sorted(sorted(numlist, reverse=True), key=lambda x : abs(n-x))
    return ls
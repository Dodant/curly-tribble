def solution(s):
    answer = 0
    cnt = 0
    while True:
        zeros = s.count('0')
        s = bin(len(s) - zeros).replace('0b', '')
        answer += zeros
        cnt += 1
        if s == '1': break
    return [cnt, answer]
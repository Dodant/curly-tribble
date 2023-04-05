def solution(s):
    answer = []
    for i in s.lower().split(" "):
        string = ''
        for d, t in enumerate(i):
            if (d+1) % 2 == 1: string += t.upper()
            else: string += t
        answer.append(string)
    return ' '.join(answer)
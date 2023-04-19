def solution(ingredient):
    answer = 0
    if len(ingredient) < 4:
        return 0
    g = ingredient[:3]
    for i in ingredient[3:]:
        g.append(i)
        if g[-4:] == [1, 2, 3, 1]:
            g.pop()
            g.pop()
            g.pop()
            g.pop()
            answer += 1
    return answer
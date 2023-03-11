def solution(sides):
    answer = 0
    for i in range(1, sum(sides)):
        side_copy = sides.copy()
        side_copy.append(i)
        if max(side_copy) < sum(side_copy) - max(side_copy):
            answer += 1
    return answer
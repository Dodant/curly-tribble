def solution(food):
    answer = []
    for i, item in enumerate(food[1:]):
        if item % 2 == 1:
            answer.append(str(i+1) * int((item-1)/2))
        else:
            answer.append(str(i+1) * int(item/2))
    return ''.join(answer) + '0' + ''.join(answer[::-1])
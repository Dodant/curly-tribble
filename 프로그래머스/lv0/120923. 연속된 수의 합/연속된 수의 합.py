def solution(num, total):
    for i in list(range(-total-100, total+100)):
        if total == sum([i for i in range(i, i+num)]):
            return [i for i in range(i, i+num)]
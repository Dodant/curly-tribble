def solution(num_list):
    time = 0
    for i in num_list:
        while True:
            if i == 1: break
            if i % 2 == 0:
                i //= 2
            elif i % 2 == 1:
                i = (i - 1) // 2
            time += 1

    return time
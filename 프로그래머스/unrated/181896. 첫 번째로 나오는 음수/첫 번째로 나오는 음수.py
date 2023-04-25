def solution(num_list):
    for i, item in enumerate(num_list):
        if item < 0:
            return i
    return -1
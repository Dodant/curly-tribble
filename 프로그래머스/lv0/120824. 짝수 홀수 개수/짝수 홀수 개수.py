def solution(num_list):
    cnt = len([i for i in num_list if i % 2 == 0])
    return [cnt, len(num_list) - cnt]
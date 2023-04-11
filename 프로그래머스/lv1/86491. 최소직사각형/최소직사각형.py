def solution(sizes):
    max_num = 0
    for i in sizes:
        if max_num < max(i):
            max_num = max(i)
    sec_max = 0
    for i in sizes:
        if sec_max < min(i):
            sec_max = min(i)
    return max_num * sec_max
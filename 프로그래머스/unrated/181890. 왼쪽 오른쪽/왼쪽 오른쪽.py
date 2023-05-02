def solution(str_list):
    for idx, item in enumerate(str_list):
        if 'l' == item:
            return str_list[:idx]
        elif 'r' == item:
            return str_list[idx+1:]
    return []
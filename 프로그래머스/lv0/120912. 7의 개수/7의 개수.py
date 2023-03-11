def solution(array):
    return len([i for i in ''.join(list(map(str, array))) if i == '7'])
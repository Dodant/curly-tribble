def solution(arr, idx):
    for i, item in enumerate(arr[idx:]):
        if item == 1:
            return idx + i
    return -1
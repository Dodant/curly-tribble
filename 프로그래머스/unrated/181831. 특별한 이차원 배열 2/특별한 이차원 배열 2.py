def solution(arr):
    answer = 0
    n = len(arr)
    for i in list(range(n)):
        for j in list(range(n)):
            if arr[i][j] != arr[j][i]:
                return 0
    return 1
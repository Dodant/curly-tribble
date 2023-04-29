def solution(arr, queries):
    for s, e, k in queries:
        for i, item in enumerate(arr):
            if s <= i <= e:
                if i % k == 0:
                    arr[i] += 1
    return arr
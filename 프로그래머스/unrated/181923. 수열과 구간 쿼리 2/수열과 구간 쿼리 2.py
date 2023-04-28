def solution(arr, queries):
    answer = []
    for s, e, k in queries:
        ls = list(filter(lambda x: x > k, arr[s:e+1]))
        if ls:
            answer.append(min(ls))
        else:
            answer.append(-1)
    return answer
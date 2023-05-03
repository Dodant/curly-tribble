def solution(arr, k):
    answer = []
    for i in arr:
        if i not in answer:
            if len(answer) == k:
                return answer
            answer.append(i)
    return answer + [-1] * (k - len(answer))
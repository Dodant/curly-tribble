def solution(arr, n):
    answer = []
    if len(arr) % 2 == 0:
        for i, item in enumerate(arr):
            if i % 2 == 1:
                answer.append(item + n)
            else:
                answer.append(item)
    elif len(arr) % 2 == 1:
        for i, item in enumerate(arr):
            if i % 2 == 0:
                answer.append(item + n)
            else:
                answer.append(item)
    return answer
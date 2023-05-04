def solution(arr):
    answer = []
    i = 0
    while True:
        if i == len(arr):
            break
        if answer == []:
            answer.append(arr[i])
            i += 1
        elif answer[-1] == arr[i]:
            answer = answer[:-1]
            i += 1
        else:
            answer.append(arr[i])
            i += 1
    if answer == []: return [-1]
    return answer
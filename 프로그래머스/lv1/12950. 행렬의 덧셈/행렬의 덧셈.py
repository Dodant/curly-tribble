def sum_(arr1, arr2):
    answer = []
    for a, b in zip(arr1, arr2):
        answer.append(a+b)
    return answer

def solution(arr1, arr2):
    answer = []
    for i in range(len(arr1)):
        answer.append(sum_(arr1[i], arr2[i]))
    return answer
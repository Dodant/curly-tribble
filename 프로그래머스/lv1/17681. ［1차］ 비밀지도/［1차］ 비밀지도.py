def solution(n, arr1, arr2):
    answer = []
    map_ = []
    for i in range(n):
        map_.append(bin(arr1[i] | arr2[i])[2:].zfill(n))
    for i in range(n):
        answer.append(map_[i].replace('1', '#').replace('0', ' '))
    return answer

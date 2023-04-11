def solution(number):
    answer = 0
    number = sorted(number)
    for i, item_i in enumerate(number):
        for j, item_j in enumerate(number[i+1:]):
            j = j + i + 1
            for k, item_k in enumerate(number[j+1:]):
                k = k + j + 1
                if i == j or i == k or j == k: continue
                if number[i] + number[j] + number[k] == 0: 
                    answer += 1
    return answer
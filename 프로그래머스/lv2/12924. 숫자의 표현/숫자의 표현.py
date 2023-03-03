def solution(n):
    answer = 0
    for i in range(1, n+1):
        cont = 0
        for j in range(i, n+1):
            cont += j
            if cont == n:
                answer += 1
                break
            if cont > n: break
    return answer
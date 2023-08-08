def solution(n):
    answer = []
    base = [[0] * n for _ in range(n)]
    
    number = 1
    x_ind, y_ind = 0, 0
    while True:
        # 1. x축 채우기
        while True:
            if x_ind == n or base[x_ind][y_ind] != 0:
                x_ind -= 1
                y_ind += 1
                break
            base[x_ind][y_ind] = number
            number += 1
            x_ind += 1
        
        # 2. y축 채우기
        while True:
            if y_ind == n or base[x_ind][y_ind] != 0:
                x_ind -= 1
                y_ind -= 2
                break
            base[x_ind][y_ind] = number
            number += 1
            y_ind += 1

            
        # 3. 대각선 채우기
        while True:
            if base[x_ind][y_ind] != 0 :
                x_ind += 2
                y_ind += 1
                break
            base[x_ind][y_ind] = number
            number += 1
            x_ind -= 1
            y_ind -= 1

        if number > (n * (n + 1)) // 2:
            break
        
    for i in range(n):
        for j in range(n):
            if base[i][j] != 0:
                answer.append(base[i][j])

    return answer
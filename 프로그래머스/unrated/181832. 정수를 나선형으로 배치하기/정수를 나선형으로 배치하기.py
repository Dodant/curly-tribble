import pprint

def solution(n):
    answer = [[0]*n for _ in range(n)]
    direction = {'r': [0, 1], 'd': [1, 0], 'l': [0, -1], 'u': [-1, 0]}
    i, j = 0, 0
    num = 1
    point = 'r'
    
    while True:
        answer[i][j] = num
        if point == 'r':
            if j == n - 1:
                point = 'd'
            elif answer[i][j+1] != 0:
                point = 'd'
            else:
                i, j = i + direction[point][0], j + direction[point][1]
                num += 1
        elif point == 'd':
            if i == n - 1:
                point = 'l'
                print('ads')
            elif answer[i+1][j] != 0:
                point = 'l'
            else:
                i, j = i + direction[point][0], j + direction[point][1]
                num += 1
        elif point == 'l':
            if j == 0:
                point = 'u'
            elif answer[i][j-1] != 0:
                point = 'u'
            else:
                i, j = i + direction[point][0], j + direction[point][1]
                num += 1
        elif point == 'u':
            if i == 0:
                point = 'r'
            elif answer[i-1][j] != 0:
                point = 'r'
            else:
                i, j = i + direction[point][0], j + direction[point][1]
                num += 1

        if num == n ** 2: 
            answer[i][j] = num
            break
    return answer
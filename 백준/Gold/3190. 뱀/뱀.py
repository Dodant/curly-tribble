N = int(input())
K = int(input())

boards = [[0]*(N+2) for _ in range(N+2)]
apples = []
transs = []


for _ in range(K): 
    apples.append([*map(int, input().split())])

L = int(input())
for _ in range(L): 
    transs.append([*input().split()])

for i in range(0, N+2):
    boards[0][i] = 9
    boards[N+1][i] = 9
    boards[i][0] = 9
    boards[i][N+1] = 9

for i, j in apples:
    boards[i][j] = 4
    
snakes = [[1,1]]

time = 0
direction = 'R'
char = ''
dx, dy = 0, 1

while True:
    
    if len(transs) != 0 and time == int(transs[0][0]):
        char = transs.pop(0)[1]

        if direction == 'R':
            if char == 'L': 
                direction = 'U'
                dx, dy = -1, 0
            elif char == 'D': 
                direction = 'D'
                dx, dy = 1, 0
        elif direction == 'L':
            if char == 'L': 
                direction = 'D'
                dx, dy = 1, 0
            elif char == 'D': 
                direction = 'U'
                dx, dy = -1, 0
        elif direction == 'U':
            if char == 'L': 
                direction = 'L'
                dx, dy = 0, -1
            elif char == 'D': 
                direction = 'R'
                dx, dy = 0, 1
        elif direction == 'D':
            if char == 'L': 
                direction = 'R'
                dx, dy = 0, 1
            elif char == 'D': 
                direction = 'L'
                dx, dy = 0, -1
        
    head_x, head_y = snakes[0][0], snakes[0][1]
    if boards[head_x + dx][head_y + dy] in [1, 9]: 
        break
    elif [head_x + dx, head_y + dy] in snakes: 
        break
    elif boards[head_x + dx][head_y + dy] == 4:
        snakes.insert(0, [head_x + dx, head_y + dy])
        boards[head_x + dx][head_y + dy] = 0
    elif boards[head_x + dx][head_y + dy] == 0:
        snakes.insert(0, [head_x + dx, head_y + dy])
        snakes.pop()
        
    time += 1    
    
print(time + 1)
    
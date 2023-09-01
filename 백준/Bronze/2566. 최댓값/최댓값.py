space = [list(map(int, input().split())) for _ in range(9)]
    
max_ = max(sum(space, []))
print(max_)
for i in range(9):
    for j in range(9):
        if space[i][j] == max_:
            print(i+1, j+1)
            exit()
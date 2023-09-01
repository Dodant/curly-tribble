N = int(input())

paper = [[0] * 100 for i in range(100)]
for i in range(N):
    x, y = map(int, input().split())
    for j in range(x, x+10):
        for k in range(y, y+10):
            paper[j][k] = 1
            
answer = 0

for i in paper:
    answer += sum(i)
print(answer)
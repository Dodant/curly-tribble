N = int(input())
xl, yl = [], []
for i in range(N):
    x, y = map(int, input().split())
    xl.append(x)
    yl.append(y)
    
print(abs(max(xl) - min(xl)) * abs(max(yl) - min(yl)))
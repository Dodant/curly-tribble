N = int(input())

ls = [0, 1, 1]
x = 0
for i in range(3, N+1):
    ls.append(ls[i-1] + ls[i-2])
    x += 1
print(ls[N], x)
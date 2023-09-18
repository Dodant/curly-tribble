n = int(input())
answer = n
for i in range(2, n+1):
    answer += i * (n//i)
print(answer)
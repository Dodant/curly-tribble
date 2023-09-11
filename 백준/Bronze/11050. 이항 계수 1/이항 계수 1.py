N, K = map(int, input().split())
answer = 1
for i in range(N, N-K, -1):
    answer *= i
for i in range(2, K+1):
    answer //= i
print(answer)
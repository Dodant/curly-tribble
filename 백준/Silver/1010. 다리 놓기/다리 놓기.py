for _ in range(int(input())):
    N, K = map(int, input().split())
    answer = 1
    for i in range(K, K-N, -1):
        answer *= i
    for i in range(2, N+1):
        answer //= i
    print(answer)
N = int(input())
answer = 0
for i in range(N//3 + 1):
    if (N - 3 * i) % 5 == 0:
        answer = i + (N - 3 * i) // 5
        break
print(answer if answer != 0 else -1)
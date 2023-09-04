N = int(input())
i = 0
while True:
    if 3 * (i - 1) * (i - 2) <= N - 1 <= 3 * i * (i - 1):
        print(i)
        break
    i += 1
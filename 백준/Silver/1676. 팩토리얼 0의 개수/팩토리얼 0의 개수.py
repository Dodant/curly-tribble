n = int(input())

answer = 0
for i in range(1, n+1):
    if i % 5 == 0:
        for j in range(5, 0,-1):
            if i % 5 ** j == 0:
                answer += j
                break
print(answer)
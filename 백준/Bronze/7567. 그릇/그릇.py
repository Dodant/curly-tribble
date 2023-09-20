n = input()
answer = 0
for i in range(len(n)-1):
    if n[i] == n[i+1]:
        answer += 5
    else:
        answer += 10
answer += 10
print(answer)
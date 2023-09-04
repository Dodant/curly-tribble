N, B = input().split()
B = int(B)
answer = 0

for idx, i in enumerate(N[::-1]):
    if i.isdigit():
        answer += int(i) * (B ** idx)
    else:
        answer += (ord(i) - 55) * (B ** idx)
        
print(answer)
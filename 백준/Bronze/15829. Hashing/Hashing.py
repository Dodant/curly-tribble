L = int(input())
st = input()
answer = 0
for i in range(len(st)):
    answer += (ord(st[i]) - 96) * 31 ** i 
print(answer)
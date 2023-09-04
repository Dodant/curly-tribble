N, B = map(int, input().split())
answer = ''

while N != 0:
    N, r = divmod(N, B)
    
    if r < 10:
        answer += str(r)
    else:
        answer += chr(r+55)
        
print(answer[::-1])
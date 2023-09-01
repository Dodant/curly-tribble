answer = 0

N = int(input())
for i in range(N):
    a = input()
    dct = {}
    for i in a:
        if i not in dct:
            dct[i] = 1
        else:
            dct[i] += 1
            
    for k, v in dct.items():
        if k*v in a:
            continue
        else:
            answer += 1
            break
        
print(N-answer)
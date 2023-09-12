import math

N = int(input())

for _ in range(N):
    n = int(input())
    flag = False
    
    if n in [0,1,2]:
        print(2)
        continue
    
    sq = math.sqrt(n)
    check = 2
    while check <= sq: 
        if n % check == 0:  
            n += 1
            check = 2
            sq = int(math.sqrt(n))
        else: 
            check += 1
    print(n)
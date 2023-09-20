from collections import Counter
import sys
input = sys.stdin.readline

def myround(n):
    if n < 0:
        if n - int(n) <= 0.5:
            return int(n - 0.5)
        else:
            return int(n)
    if n - int(n) >= 0.5:
        return int(n + 0.5)
    else:
        return int(n)
    

n = int(input())
ls = []

for _ in range(n):
    ls.append(int(input().strip()))
             
ls.sort()
bm = Counter(ls).most_common()
cmn = 0

if len(bm) == 1:
    cmn = bm[0][0]
else:
    if bm[0][1] == bm[1][1]:
        cmn = bm[1][0]
    else:
        cmn = bm[0][0]

print(myround(sum(ls)/n))
print(ls[n//2])
print(cmn)
print(max(ls) - min(ls))
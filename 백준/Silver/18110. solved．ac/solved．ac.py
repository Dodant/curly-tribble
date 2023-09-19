import sys 
input = sys.stdin.readline

def my_round(n):
    if n - int(n) >= 0.5:
        return int(n) + 1
    else:
        return int(n)


n = int(input().strip())
ls = []

if n == 0:
    print(0)
else:
    for i in range(n):
        ls.append(int(input().strip()))
    ls.sort()
    t = my_round(n*0.15 + 0.00005)
    ls = ls[t:n-t]
    print(my_round(sum(ls)/len(ls)))
    
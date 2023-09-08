N, k = map(int, input().split())
ls = list(map(int, input().split()))
ls.sort()   
print(ls[-k])
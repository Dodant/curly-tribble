N = int(input())
ls = list(map(int, input().split()))
st = list(set(ls))
st.sort()
fc = {}

for idx, item in enumerate(st):
    fc[item] = idx
    
answer = [0 for _ in range(N)]
for i in range(len(ls)):
    answer[i] = fc[ls[i]]
print(*answer)
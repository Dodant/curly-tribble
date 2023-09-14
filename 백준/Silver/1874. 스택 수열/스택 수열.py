n = int(input())
ls = [int(input()) for _ in range(n)]

st = []
point = 0
nums = [i for i in range(1, n+1)]
answer = []
for i in nums:
    st.append(i)
    answer.append('+')
    while st and st[-1] == ls[point]:
        st.pop()
        answer.append('-')
        point += 1
        if point == n: break
        
if st: print('NO')
else: print('\n'.join(answer))
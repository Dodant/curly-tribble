T = input()
T = T.replace('()', '|')

st = []
answer = 0

for i in T:
    if i == '(':
        st.append(i)
    elif i == '|':
        answer += len(st)
    elif i == ')':
        st.pop()
        answer += 1
        
print(answer)
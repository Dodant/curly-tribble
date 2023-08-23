N = int(input())
tar1, tar2 = map(int, input().split())
M = int(input())
relation = [[*map(int, input().split())] for _ in range(M)]
up = dict()

for i in range(N): up[i+1] = []
for f, s in relation: up[s] = up.get(s, []) + [f]

tar1_st, tar1_p, tar1_d = [tar1], [tar1], [0]
tar2_st, tar2_p, tar2_d = [tar2], [tar2], [0]

while tar1_st:
    p = tar1_st.pop(0)
    if up[p] == []: break
    tar1_p.append(up[p][0])
    tar1_st.append(up[p][0])
    tar1_d.append(tar1_d[-1] + 1)

while tar2_st:
    p = tar2_st.pop(0)
    if up[p] == []: break
    tar2_p.append(up[p][0])
    tar2_st.append(up[p][0])
    tar2_d.append(tar2_d[-1] + 1)

t = -1

for i in tar1_p:
    if i in tar2_p:
        t = tar1_d[tar1_p.index(i)] + tar2_d[tar2_p.index(i)]
        break
print(t)
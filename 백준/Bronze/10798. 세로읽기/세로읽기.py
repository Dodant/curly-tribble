space = [input() for _ in range(5)]
max_len = max([len(i) for i in space])

for i in range(5):
    if len(space[i]) < max_len:
        space[i] = space[i] + ' ' * (max_len - len(space[i]))

answer = ''
for i in zip(*space):
    for j in i:
        answer += j
print(answer.replace(' ', ''))
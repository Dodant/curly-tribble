answer = []
def hanoi(n, start, to, mid, answer):
    if n == 1: return answer.append([start, to])
    hanoi(n-1, start, mid, to, answer)
    answer.append([start, to])
    hanoi(n-1, mid, to, start, answer)

hanoi(int(input()), 1, 3, 2, answer)
print(len(answer))
for i in answer:
    print(*i)
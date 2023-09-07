N = int(input())
ls = [*map(int, input().split())]
print(len([i for i in ls if i != 1 and len([j for j in range(1, i+1) if i % j == 0]) == 2]))
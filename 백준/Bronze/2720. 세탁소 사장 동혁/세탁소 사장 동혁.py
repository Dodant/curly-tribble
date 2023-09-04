for i in range(int(input())):
    C = int(input())
    R, V = C // 25, C % 25
    print(R, end=' ')
    R, V = V // 10, V % 10
    print(R, end=' ')
    R, V = V // 5, V % 5
    print(R, end=' ')
    print(V, end=' ')
    print()
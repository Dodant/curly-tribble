import heapq 

T = int(input())

for _ in range(T):
    n = int(input())
    
    T1 = []
    for _ in range(n//10+1):
        T1 += list(map(int, input().split()))
        
    left, median, right = [], T1[0], []
    answer = [T1[0]]
    
    i = 1
    for _ in range(len(T1)//2):
        for _ in range(2):
            if T1[i] < median:
                heapq.heappush(left, -T1[i])
            else:
                heapq.heappush(right, T1[i])
            i += 1
            
        while len(left) != len(right):
            if len(left) > len(right):
                v = -heapq.heappop(left)
                heapq.heappush(right, median)
                median = v
            else:
                v = heapq.heappop(right)
                heapq.heappush(left, -median)
                median = v
        answer.append(median)
    
    print(len(T1) // 2 + 1)
    for i in range(len(answer) // 10 + 1):
        print(' '.join(list(map(str, answer[i*10:i*10+10]))))
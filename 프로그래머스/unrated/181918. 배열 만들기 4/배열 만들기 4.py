def solution(arr):
    stk = []
    i = 0
    while True:
        if stk == []:
            stk.append(arr[i])
            i += 1
        elif stk[-1] < arr[i]:
            stk.append(arr[i])
            i += 1
        else:
            stk.pop()
        
        if i == len(arr):
            break
    return stk
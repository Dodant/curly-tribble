def solution(s):
    stack = []
    for i in s:
        if i == '(': stack.append(0)
        if i == ')':
            if not stack: return False
            stack.pop()
    if not stack: return True
    return False
def solution(myString, pat):
    answer = set()
    for i in range(len(myString)):
        if myString[i:i+len(pat)] == pat:
            answer.add(i)
    return len(answer)
def solution(triangle):

    height = len(triangle)
    dp = [[0] * height for i in range(1, height+1)]
    dp[0][0] = triangle[0][0]

    for i in range(1, height):
        for j in range(i+1):
            dp[i][j] = max(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j]

    return max(dp[-1])

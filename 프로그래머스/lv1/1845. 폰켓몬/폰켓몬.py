def solution(nums):
    lmt = len(nums) // 2
    nums = list(set(nums))
    return len(nums) if len(nums) < lmt else lmt
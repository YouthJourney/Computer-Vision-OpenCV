# -*- coding: UTF-8 -*-    
# Author: LGD
# FileName: test
# DateTime: 2020/10/18 21:11 
# SoftWare: PyCharm
from typing import List


def two_sum(nums: List[int], target: int):
    len1 = len(nums)
    res = []
    for i in range(len1):
        for n in range(i + 1, len1):
            if nums[i] + nums[n] == target:
                res.append(i)
                res.append(n)
                break
    return res


if __name__ == '__main__':
    nums1 = [2, 7, 11, 15]
    target1 = 9

    print(two_sum(nums1, target1))

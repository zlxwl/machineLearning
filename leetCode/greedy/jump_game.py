from typing import List


# class Solution:
#     def canJump(self, nums: List[int]) -> bool:
#         flag = True
#         index = 0
#
#         while flag and index < len(nums):
#             index = index + nums[index]
#             if index >= len(nums) -1 :
#                 return True
#             elif nums[index] == 0 and index < len(nums) and sum(nums[0 :index]) < len(nums):
#                 return False
# #         return False
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        length = len(nums)
        min_jump_dist = 1
        for i in range(length-2, -1, -1):
            if nums[i] >= min_jump_dist:
                min_jump_dist = 1
            else:
                min_jump_dist += 1
            if i == 0 and min_jump_dist > 1:
                return False
        return True



if __name__ == "__main__":
    print(Solution().canJump([3, 2, 1, 0, 4]))
    print(Solution().canJump([0]))

    # for i in range(6,1,-1):
    #     print(str(i))
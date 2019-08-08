from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for index in range(1, len(prices)):
            if prices[index] > prices[index-1]:
                profit += prices[index] - prices[index-1]

        return profit


if __name__ == "__main__":
    # print(Solution().maxProfit([7,1,5,3,6,4]))
    # print(Solution().maxProfit([1,2,3,4,5]))
    print(Solution().maxProfit([6, 1, 3, 2, 4, 7]))


class Solution:
    def is_theses(self, token):
        if token == "(" or token == ")":
            return True
        return False


    def removeOuterParentheses(self, S: str) -> str:
        res_list = []
        stack_list = []
        for i in stack_list:
            if i == "(":
                stack_list.append("(")
            if stack_list:
                res_list.append(i)
            if i == ")":
                stack_list.pop()
        return  "".join(res_list)


if __name__ == "__main__":
    a ='(()())(())'
    print("data")
    b = Solution().removeOuterParentheses(a)
    print(b)
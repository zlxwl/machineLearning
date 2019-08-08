from typing import List
class PrintNum(object):
    def __init__(self, num1, num2, num3):
        self.num1 = num1
        self.num2 = num2
        self.num3 = num3

    def print_num(self):
        print(self.num1,  self.num2, self.num3)

    @classmethod
    def list_num(cls, list_num: List):
        """
        :type list_num: List
        """
        return (cls(list_num[0], list_num[1], list_num[2]))

if __name__ == '__main__':
    a = PrintNum(1, 2, 3)
    # a.print_num()
    b = PrintNum.list_num([1, 2, 3, 44])
    b.print_num()
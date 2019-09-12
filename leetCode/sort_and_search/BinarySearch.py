

def binary_search(array, number):
    middle = int(len(array)/2)
    if len(array) == 1 and array[middle] != number:
        print("not found")
        return None
    elif array[middle] == number:
        print("find number")
        return middle
    elif number > array[middle]:
        binary_search(array[middle+1:], number)
    elif number < array[middle]:
        binary_search(array[:middle], number)

if __name__ == "__main__":
    x = [i for i in range(0, 100, 3)]
    print(binary_search(x, 69))



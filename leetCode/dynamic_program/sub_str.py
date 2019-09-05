import numpy as np

def find_max_sub_str(string_a, string_b):
    cell = np.zeros((len(string_a), len(string_b)))

    index_a = 0
    for i in range(len(string_a)):
        if string_a[i] == string_b[i]:
            cell[i, 0] == 1
            index_a = i
            break
    for i in range(index_a, len(string_a)):
        cell[i, 0] = 1

    index_b = 0
    for i in range(len(string_b)):
        if string_b[i] == string_a[i]:
            cell[0, i] == 1
            index_b
            break
    for j in range(index_b, len(string_b)):
        cell[0, j] = 1

    for i in range(1, len(string_a)):
        for j in range(1, len(string_b)):
            if string_a[i] == string_b[j]:
                cell[i, j] = cell[i-1, j-1] + 1
            else:
                cell[i, j] = max(cell[i-1, j], cell[i, j-1])

    max_len = 0
    for i in range(0, len(string_a)):
        for j in range(0, len(string_b)):
            if cell[i, j] > max_len:
                max_len = cell[i, j]

    return max_len


if __name__ == "__main__":
    print(find_max_sub_str("fish", "fosh"))

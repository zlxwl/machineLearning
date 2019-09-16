import random


# 使用list模拟栈
def search(name):
    def is_seller(name):
        return "e" in name

    searched_list = []
    search_stack = []
    search_stack.append(graphy["you"])
    while search_stack:
        node = search_stack.pop()
        if node not in searched_list:
            if is_seller(node):
                print(node)
                searched_list.append(node)
                return True
            else:
                child_nodes = graphy[node]
                child_node = random.choice(child_nodes)
                search_stack.append(child_node)
    return False


if __name__ == "__main__":
    graphy = {}
    graphy["you"] = ["alice", "bob", "claire"]
    graphy["bob"] = ["anuj", "peggy"]
    graphy["alice"] = ["peggy"]
    graphy["claire"] = ["thom"]
    graphy["aunj"] = []
    graphy["thon"] = []
    graphy["jonny"] = []
    graphy["peggy"] = []
    # search("you")
    print(random.choice(["aa", "bb", "cc"]))
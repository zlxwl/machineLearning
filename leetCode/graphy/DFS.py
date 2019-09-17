import random


def is_seller(name):
    return "e" in name


def search(graphy, name):
    searched_nodes = []
    search_stacks = []
    search_stacks.append(name)
    while search_stacks:
        neighbor_node = random.choice(graphy[name])
        search_stacks.append(neighbor_node)
        if neighbor_node not in searched_nodes and is_seller(neighbor_node):
            print(neighbor_node)
            searched_nodes.append(neighbor_node)
            return True
        elif graphy[neighbor_node]:
            child = random.choice(graphy[neighbor_node])
            if child and child not in searched_nodes:
                search_stacks.append(child)
                searched_nodes.append(child)
        else:
            search_stacks.pop()
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
    print(search(graphy, "you"))
    







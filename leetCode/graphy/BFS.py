from collections import deque

def search(name):

    def is_seller(name):
        return "e" in name

    searched_list = []
    search_queue = deque()
    search_queue += graphy[name]
    # search_queue +=
    while search_queue:
        person = search_queue.popleft()
        if person not in search_queue:
            if is_seller(person):
                searched_list.append(person)
                print(person)
                return True
            else:
                searched_list.append(person)
                search_queue += graphy[person]

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
    search("you")
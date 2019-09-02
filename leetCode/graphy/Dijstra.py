

def find_sortest_path(grapyh, cost, parent):
    processed = []

    def find_lowest_cost_node(precessed, cost):
        node = None
        lowest_cost = float("inf")
        for x in cost.keys():
            if x not in precessed and cost[x] < lowest_cost:
                node = x
                lowest_cost = cost[x]
        return node

    node = find_lowest_cost_node(processed, cost)
    while node:
        cos = cost[node]
        neibor_nodes = graphy[node]
        for n in neibor_nodes.keys():
            if cos + neibor_nodes[n] < cost[n]:
                parent[n] = node
                cost[n] = cos + neibor_nodes[n]
        processed.append(node)
        node = find_lowest_cost_node(processed, cost)
    return cost["end"]


if __name__ == "__main__":
    graphy={}
    graphy["start"] = {}
    graphy["start"]["a"] = 6
    graphy["start"]["b"] = 2
    graphy["a"] = {}
    graphy["a"]["end"] = 1
    graphy["b"] = {}
    graphy["b"]["a"] = 3
    graphy["b"]["end"] = 5
    graphy["end"] = {}

    cost={}
    cost["a"] = 6
    cost["b"] = 2
    cost["end"] = float("inf")

    parent={}
    parent["a"] = "start"
    parent["b"] = "start"
    parent["end"] = None

    print(find_sortest_path(graphy, cost, parent))


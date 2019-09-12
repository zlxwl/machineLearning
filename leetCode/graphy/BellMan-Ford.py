
def bellman_ford(grapy):
    vertexs = grapy.keys()
    distance = {}
    process_vex = {}
    # init
    for index, vertex in enumerate(vertexs):
        if index == 0:
            distance[vertex] = 0
        else:
            distance[vertex] = float("inf")

    # loop
    for _ in range(len(vertexs)-1):
        for v in vertexs:
            neighbors = grapy[v].keys()
            for node in neighbors:
                new_distance = distance[v] + grapy[v][node]
                if new_distance < distance[node]:
                    distance[node] = new_distance
                    process_vex[node] = v

    # check
    for v in vertexs:
        neighbors = grapy[v].keys()
        for node in neighbors:
            new_distance = grapy[v][node]
            if distance[v] + new_distance < distance[node]:
               print("has negative loop")
               break

    return distance, process_vex


if __name__ == "__main__":
    grapy = {}
    grapy["start"] = {}
    grapy["start"]["E"] = 8
    grapy["start"]["A"] = 10
    grapy["A"] ={}
    grapy["A"]["C"] = 2
    grapy["E"] = {}
    grapy["E"]["D"] = 1
    grapy["D"] = {}
    grapy["D"]["C"] = -1
    grapy["D"]["A"] = -4
    grapy["C"] = {}
    grapy["C"]["B"] = -2
    grapy["B"] = {}
    grapy["B"]["A"] = 1
    print(bellman_ford(grapy))
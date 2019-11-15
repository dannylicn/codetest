from queue import Queue
from graph import *

def bfs(graph, start=0):
    queue = Queue()
    queue.put(start)

    visited = np.zeros(graph.numVertices)

    while not queue.empty():
        vertex = queue.get()

        if visited[vertex] == 1:
            continue

        print("Visit:", vertex)
        visited[vertex] = 1

        for v in graph.get_adjacent_vertices(vertex):
            if visited[v] != 1:
                queue.put(v)


def dfs(graph, visited, current=0):
    if visited[current] == 1:
        return

    visited[current] = 1
    print("Visit:", current)
    for vertex in graph.get_adjacent_vertices(current):
        dfs(graph, visited, vertex)

# g = AdjacencyGraphMatrix(9)

# g.add_edge(0,1)
# g.add_edge(1,2)
# g.add_edge(2,7)
# g.add_edge(2,4)
# g.add_edge(2,3)
# g.add_edge(1,5)
# g.add_edge(5,6)
# g.add_edge(6,3)
# g.add_edge(3,4)
# g.add_edge(6,8)

# # bfs(g,0)
# visited = np.zeros(g.numVertices)
# dfs(g,visited)

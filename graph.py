import abc;
import numpy as np;

class Graph(abc.ABC):

    def __init__(self, numVertices, directed=False):
        self.numVertices = numVertices
        self.directed = directed

    @abc.abstractmethod
    def add_edge(self, v1, v2, weight=0):
        pass

    @abc.abstractmethod
    def get_adjacent_vertices(self, v):
        raise NotImplementedError

    @abc.abstractmethod
    def get_indegree(self, v):
        raise NotImplementedError

    @abc.abstractmethod
    def get_edge_weight(self, v1, v2):
        raise NotImplementedError

    @abc.abstractmethod
    def display(self):
        raise NotImplementedError


class AdjacencyGraphMatrix(Graph):
    def __init__(self, numVertices, directed=False):
        super(AdjacencyGraphMatrix,self).__init__(numVertices, directed)
        self.adjacencyMatrix = np.zeros((numVertices,numVertices))

    def add_edge(self, v1, v2, weight=1):
        if(v1 < 0 or v2 < 0 or v1==v2 or v1>=self.numVertices or v2 >= self.numVertices):
            raise ValueError("Vertex v1 %d  v2 %d cannot be edge " % (v1,v2))

        self.adjacencyMatrix[v1][v2] = weight
        if self.directed==False:
            self.adjacencyMatrix[v2][v1] = weight


    def get_adjacent_vertices(self, v):
        if v >= self.numVertices or v<0:
            raise ValueError("Vertex %d is invalid" % v)

        adjacent_vertices = []
        for i in range(self.numVertices):
            if self.adjacencyMatrix[v][i]>0:
                adjacent_vertices.append(i)
        
        return adjacent_vertices

    def get_indegree(self, v):
        if v >= self.numVertices or v<0:
            raise ValueError("Vertex %d is invalid" % v)
        indegree = 0
        for i in range(self.numVertices):
            if self.adjacencyMatrix[i][v]>0:
                indegree = indegree+1

        return indegree

    def get_edge_weight(self, v1, v2):
        return self.adjacencyMatrix[v1][v2]

    def display(self):
        for i in range(self.numVertices):
            for j in self.get_adjacent_vertices(i):
                print("edge %d --> %d" % (i, j))

class Node:
    def __init__(self, vertexId):
        self.vertexId = vertexId
        self.adjacency_dict = {}

    def add_edge(self, v, weight=1):
        if self.vertexId == v:
            raise ValueError("The vertex %d cannot be adjacent to itself" % v)
        self.adjacency_dict[v] = weight

    def get_adjacent_vertices(self):
        return sorted(self.adjacency_dict.keys())

    def get_weight(self, v):
       return self.adjacency_dict.get(v)



class AdjacencyGraphSet(Graph):
    def __init__(self, numVertices, directed=False):
        super(AdjacencyGraphSet,self).__init__(numVertices, directed)
        self.vertex_list = []
        for i in range(numVertices):
            self.vertex_list.append(Node(i))

    def add_edge(self, v1, v2, weight=1):
        if(v1 < 0 or v2 < 0 or v1==v2 or v1>=self.numVertices or v2 >= self.numVertices):
            raise ValueError("Vertex v1 %d  v2 %d cannot be edge " % (v1,v2))
        # if (weight!=1):
        #     raise ValueError("weight only can be 1")

        self.vertex_list[v1].add_edge(v2, weight)
        if self.directed == False:
            self.vertex_list[v2].add_edge(v1,weight)

    def get_adjacent_vertices(self, v):
        if v >= self.numVertices or v<0:
            raise ValueError("Vertex %d is invalid" % v)

        return self.vertex_list[v].get_adjacent_vertices()

    def get_indegree(self, v):
        if v >= self.numVertices or v<0:
            raise ValueError("Vertex %d is invalid" % v)
        indegree = 0
        for i in range(self.numVertices):
            if v in self.get_adjacent_vertices(i):
                indegree = indegree + 1

    def get_edge_weight(self, v1, v2):
        return self.vertex_list[v1].get_weight(v2)

    
    def display(self):
        for i in range(self.numVertices):
            for j in self.get_adjacent_vertices(i):
                print("edge %d --> %d" % (i, j))
    


# adjacencyMatrix = AdjacencyGraphMatrix(6, directed=True)

# adjacencyMatrix.add_edge(0, 1)
# adjacencyMatrix.add_edge(1, 2)
# adjacencyMatrix.add_edge(1, 3)
# adjacencyMatrix.add_edge(2, 4)
# adjacencyMatrix.add_edge(3, 4)
# adjacencyMatrix.add_edge(2, 5)
# adjacencyMatrix.add_edge(4, 5)

# adjacencyMatrix.display()

# adjacencySet = AdjacencyGraphSet(6, directed=False)

# adjacencySet.add_edge(0, 1)
# adjacencySet.add_edge(1, 2)
# adjacencySet.add_edge(1, 3)
# adjacencySet.add_edge(2, 4)
# adjacencySet.add_edge(3, 4)
# adjacencySet.add_edge(2, 5)
# adjacencySet.add_edge(4, 5)

# adjacencySet.display()



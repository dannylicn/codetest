# !usr/bin/python3
"""
Different graph algorithms implementation
Arash Tehrani
"""
class Graph(object):
    def __init__(self, neighbor_incidence):
        """
        input:
            _ neighbor_incidence: dictionary with kes being indices of the ndoes and
                    values being list of their corresponding neighbors
        """
        self.adj = neighbor_incidence
        self.num_nodes = len(self.adj)

    def make_undirected(self, weights):
        """
        given the directed edges in adj, build an undirected weighted version of the 
        graph which transform directed edges to undirected ones
        inputs:
            - weights: symmetric 2d array of undirected weights     
        """
        self.undirected_adj = {k:v[:] for k,v in self.adj.items()}
        for u in range(self.num_nodes):
            for v in self.adj[u]:
                if u not in self.undirected_adj[v]:
                    self.undirected_adj[v].append(u)
        
        self.undirected_w = {}
        for u in range(self.num_nodes):
            for v in self.undirected_adj[u]:
                self.undirected_w.setdefault(u, []).append(weights[u, v])
        
    def cycle_existence(self):
        """
        check if a cycle exists in the directed graph and outputs the cycle
        Trick: Run DFZS and see if you are gonna visit already visited node in the 
        same run 
        inputs:
            -
        outputs:
            cycle: list of ints, index of the nodes in the cycle, if no cycle 
                    exists, returns None
        """
        visited_set = [False]*self.num_nodes
        parent = {}
        
        def dfs_visit(u, visited_set, current_set):
            """
            perform one visit of depth first search
            """
            visited_set[u] = True
            current_set.append(u)
            
            for v in self.adj[u]:
                if v in current_set:
                    cycle = [v,u]
                    idx = u
                    while parent[idx] is not None and v != parent[idx]:
                        idx = parent[idx]
                        cycle.append(idx)
                    return cycle

                elif visited_set[v] == False:
                    parent[v] = u
                    cycle = dfs_visit(v, visited_set, current_set)
                    if cycle:
                        return cycle
            else:
                #print('before removing:', u, self.adj[u])
                current_set.remove(u)
            
            return None

        for i in range(self.num_nodes):
            if visited_set[i] == False:
                parent[i] = None
                cycle = dfs_visit(i, visited_set, [])
                if cycle:
                    return cycle
        return None  
    
    def topological_sort(self):
        """
        runs topological sort on the graph
        inputs:
            -
        outputs:
            stack: list oof ints, returns the stack containing the result
        """   
        if g.cycle_existence():
            print('Topological does not exist')
            return None
            
        visited_set = [False]*self.num_nodes
        stack = []

        def TSvisit(u, visited_set, stack):
            visited_set[u] = True
            for v in self.adj[u]:
                if visited_set[v] == False:
                    TSvisit(v, visited_set, stack)
            else:
                stack.append(u)

        for i in range(self.num_nodes):
            if visited_set[i] == False:
                TSvisit(i, visited_set, stack)
        
        return stack

    def bfs(self, node_idx):
        """
        performs the breadth first search on rooted at node_idx
        input:
            - node_idx: int, index of the root node
        output:
            - node_level: dictionary, keys are indices of the nodes and 
                    values are their corresponding levels in bfs tree
            - level: dictionary, keys are levels and values are the nodes 
                    in each level in bfs tree
            - parent: dictionary, keys are indices of the ndoes and values are
                    their parent in bfs tree
        """
        node_level = {node_idx: 0}
        parent = {node_idx: None}
        i = 1
        frontier = [node_idx]
        while frontier:
            next_frontier = []
            for u in frontier:
                for v in self.adj[u]:
                    if v not in node_level:
                        node_level[v] = i
                        parent[v] = u
                        next_frontier.append(v)
            frontier = next_frontier
            i += 1
        
        level = {}
        for j in range(i-1):
            level[j] = []
        for i,j in node_level.items():
            level[j].append(i)

        return level, node_level, parent

    def dfs(self, node_idx):
        """
        performs the depth first search on rooted at node_idx
        input:
            - node_idx: int, index of the root node
        output:
            - parent: dictionary, keys are indices of the ndoes and values are
                    their parent in bfs tree
            - search_path: list if indices, order in which dfs traverse the graph
        """
        parent = {node_idx: None}
        
        def dfs_visit(u, parent, path):
            for v in self.adj[u]:
                if v not in parent:
                    parent[v] = u
                    path.append(v)
                    dfs_visit(v, parent, path)

        search_path = [node_idx]
        dfs_visit(node_idx, parent, search_path)
        
        return parent, search_path

    def bellman_ford(self, node_idx, weights):
        """
        Bellman_Ford algorithm with negative weight cycle detection
        inputs:
            - node_idx: int, index of the root node
            - weights: dictionaru with keys being the indices of the ndoes and values 
                    being the list of outgoing edges
        outputs:
            - dist: list, list of distances for nodes to root
            - parent: dictionary, keys are indices of the ndoes and values are
                    their parent in bfs tree
        """
        parent = {node_idx: None}
        dist = [self.num_nodes+1]*self.num_nodes
        dist[node_idx] = 0

        for _ in range(self.num_nodes-1):
            for u in range(self.num_nodes):
                for iv, v in enumerate(self.adj[u]):
                    if dist[v] > dist[u] + weights[u][iv]:
                        dist[v] = dist[u] + weights[u][iv]
                        parent[v] = u

        # detecting negative cycle existance
        for u in range(self.num_nodes):
            for iv, v in enumerate(self.adj[u]):
                if dist[v] > dist[u] + weights[u][iv]:
                    print('Negative cycle exists')
                    return None, None

        return dist, parent

    def djikstra(self, node_idx, weights):
        """
        Djikstra shortest path algorithms
        inputs:
            - node_idx: int, index of the root node
            - weights: dictionaru with keys being the indices of the ndoes and values 
                    being the list of outgoing edges
        outputs:
            - dist: list, list of distances for nodes to root
            - parent: dictionary, keys are indices of the ndoes and values are
                    their parent in bfs tree
        """
        parent = {node_idx: None}
        dist = [self.num_nodes+1]*self.num_nodes
        dist[node_idx] = 0

        # better implementation would be by using a priority queue
        queue = [True]*self.num_nodes
        queue_dist = [[i,dist[i]] for i,q in enumerate(queue) if q]
        while queue_dist:
            idx = min(queue_dist, key=lambda x:x[1])
            u = idx[0]
            queue[idx[0]] = False
            for iv, v in enumerate(self.adj[u]):
                if dist[v] > dist[u] + weights[u][iv]:
                    dist[v] = dist[u] + weights[u][iv]
                    parent[v] = u
            queue_dist = [[i,dist[i]] for i,q in enumerate(queue) if q]
            
        return dist, parent
    
    def kruskal_mst(self):
        """
        Kruskal's minimum spanning tree algorithm
        inputs:
            -
        outputs:
            - mst_edges: list of ints, list of mst edges
        """
        # we strat by ordering the edges based on their weights
        edge_weights = []
        edge_counter = 0

        for u in range(self.num_nodes):
            for iv, v in enumerate(self.undirected_adj[u]):
                if u < v:
                    edge_weights.append(((u,v), self.undirected_w[u][iv]))
                    edge_counter += 1
                    
        edge_weights.sort(key = lambda x: x[1])
        # making disjoint sets of the nodes
        # i am implementing disjoint sets through two hashes
        # note: it can join the previous for but is re-done here for clarity
        disjoint_sets = {}
        set_pointer = {}
        for u in range(self.num_nodes):
            disjoint_sets[u] = {u}  #key: index of the set, val: the set
            set_pointer[u] = u  # key: index of the node, val: index of the set
        
        mst_edges= []
        for ew in edge_weights:
            e = ew[0]
            p1, p2 = set_pointer[e[0]], set_pointer[e[1]]
            if p1 != p2:
                new_set = disjoint_sets[p1] | disjoint_sets[p2]
                new_pointer_idx = min(p1, p2)
                
                del disjoint_sets[p1]
                del disjoint_sets[p2]

                for v in new_set:
                    set_pointer[v] = new_pointer_idx
                disjoint_sets[new_pointer_idx] = new_set

                mst_edges.append(e)
            
            #print('disjoint_set:', disjoint_sets)
            if len(disjoint_sets) == 1:
                break
            
        return mst_edges

    def prim_mst(self):
        """
        Prim's minimum spanning tree algorithm
        inputs:
            -
        outputs:
            - mst_edges: list of ints, list of mst edges
        """
        node_array = [float('inf')]*self.num_nodes
        mst_edges = [(0,0)]*self.num_nodes
        node_array[0] = 0
        
        node_list = dict.fromkeys(range(self.num_nodes), True)
        
        queue = [v for v in enumerate(node_array) if node_list[v[0]]]
        while queue:
            u = min(queue, key = lambda x: x[1])[0]
            for i, v in enumerate(self.undirected_adj[u]):
                if node_list[v] and self.undirected_w[u][i] < node_array[v]:
                    node_array[v] = self.undirected_w[u][i]
                    mst_edges[v] = (u,v)
            else:
                node_list[u] = False
            queue = [v for v in enumerate(node_array) if node_list[v[0]]]
        
        return mst_edges[1:]

    def floyd_warshall(self, weights):
        """
        Run Floys_Warshall's all pairs shortest path algorithm
        inputs:
            - weights: dictionary, keys: nodes indices, values: list of outgoing edges
        outputs:
            distance_mat: list of lists, shortest weighhted path from each node to other
            path_mat: list of lists, map of shortest connection from one node to another
        """

        distance_mat = [[float('inf')]*self.num_nodes for i in range(self.num_nodes)]
        path_mat = [[self.num_nodes]*self.num_nodes for i in range(self.num_nodes)]

        for u in range(self.num_nodes):
            distance_mat[u][u] = 0
            for i, v in enumerate(self.adj[u]):
                distance_mat[u][v] = weights[u][i]
                path_mat[u][v] = u
                
        for k in range(self.num_nodes):
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    if distance_mat[u][v] > distance_mat[u][k] + distance_mat[k][v]: 
                        distance_mat[u][v] = distance_mat[u][k] + distance_mat[k][v]
                        path_mat[u][v] = path_mat[k][v]

        return distance_mat, path_mat

    def fl_reader(self, source, sink, path_mat):
        """
        reads the result of floyd_warshall function
        inputs:
            - source: int, index of the source node
            - sink: int, index of the sink node
            - path_mat: list of lists, map of shortest connection from one node to another
        outputs:
            - path: list of ints, list of nodes in the path from source to sink including 
                    source and sink
        """
        if source == sink:
            return [source]
        path = []
        u = sink
        while u != source:
            u = path_mat[source][u]
            if u == self.num_nodes:
                return None
            path.append(u)
        
        return path[::-1] + [sink]

    def crerating_weight_matrix(self, weights):
        
        weights_mat = [[0]*self.num_nodes for _ in range(self.num_nodes)]
        for u in range(self.num_nodes):
            for i, v in enumerate(self.adj[u]):
                weights_mat[u][v] = weights[u][i]
        
        return weights_mat

    def ford_fulkerson(self, source, sink, weights):
        
        weights_mat = self.crerating_weight_matrix(weights)
        
        def bfs_iteration(source, sink, w_mat):
            parent = {source: None}
            frontier = [source]
            while frontier:
                next_f = []
                for u in frontier:
                    for v, val in enumerate(w_mat[u]):
                        if val > 0 and v not in parent:
                            next_f.append(v)
                            parent[v] = u
                            if v == sink:
                                return parent
                frontier = next_f
            else:
                return None

        parent = 1 
        max_flow = 0 
        while parent is not None:
            parent = {}
            parent = bfs_iteration(source, sink, weights_mat)
            if parent is not None:
                path = [sink]
                node = sink
                min_val = float('inf')
                while node != source:
                    node = parent[node]
                    path = [node] + path
                    if weights_mat[path[-2]][path[-1]] < min_val:
                        min_val = weights_mat[path[-2]][path[-1]]
                    
                max_flow += min_val
                for i in range(len(path)-1):
                    weights_mat[path[i]][path[i+1]] -= min_val
                    weights_mat[path[i+1]][path[i]] += min_val
        
        return max_flow

#   -------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    #   -----------------------------
    # build a random directed graph or a deterministic graph
    graph_type = 'random'
    num_nodes = 20
    adj = {}
    w = {}

    if graph_type == 'random':
        for i in range(num_nodes):
            adj[i] = np.random.permutation(num_nodes)[:np.random.random_integers(num_nodes//2)]
            adj[i] = [k for k in adj[i] if k != i] # omitting self connections
            w[i] = 0.75 + 0.25*np.random.rand(len(adj[i]))
            
    elif graph_type == 'deterministic':
        for i in range(num_nodes):
            adj[i] = [(i+1) % num_nodes, 
                    (i+2) % num_nodes, 
                    (i + 3) % num_nodes]
            w[i] = adj[i]
    
    elif graph_type == 'tree':
        for i in range(num_nodes):
            if 2*i+1 < num_nodes:
                adj[i] = [2*i+1]
            else:
                adj[i] = []
            if 2*i+2 < num_nodes:
                adj[i].append(2*i+2)

            if len(adj[i]) > 0:
                w[i] = [1]*len(adj[i])
            else:
                w[i] = []

    g = Graph(adj) 
    print('adj:')
    print(adj)
    #   -----------------------------
    # Testing cycle existence
    print('------------------------------------------------')
    print('Running cycle existence')
    cycle = g.cycle_existence()
    print('nodes in cycle are:', cycle)
    if cycle is not None:
        for i in cycle:
            print(i, g.adj[i])
    #   -----------------------------
    # Testing topological sort
    print('------------------------------------------------')
    print('Running topological sort')
    stack = g.topological_sort()
    print(stack)
    #   -----------------------------
    # breadth first search
    print('------------------------------------------------')
    print('Running breadth first search')
    level, node_level, _ = g.bfs(0)
    for i, nodes in level.items():
        print('level', i, 'nodes: ', nodes)
    #   -----------------------------
    # depth first search
    print('------------------------------------------------')
    print('Running depth first search')
    parent, search_path = g.dfs(0)
    print(search_path)
    #   -----------------------------
    # Bellman_Ford shortest path algorithm
    print('------------------------------------------------')
    print('Running Bellman_Ford shortest path algorithm')
    dist, parent = g.bellman_ford(0, w)
    print(dist)
    #   -----------------------------
    # Djikstra shortest path algorithm
    print('------------------------------------------------')
    print('Running Djikstra path algorithm')
    # the weights should be positive
    dist, parent = g.djikstra(0, w)
    print(dist)
    #   -----------------------------
    # build the undirected version
    print('------------------------------------------------')
    print('Build the undirected version of the graph')
    w_mat = 0.75 + np.random.randn(num_nodes, num_nodes)
    w_mat = (w_mat + w_mat.T)/2
    g.make_undirected(w_mat)
    #for i in range(g.num_nodes):
    #    print('neighbors of node', i, ':', g.undirected_adj[i])
    #    print('corresponding weights:', g.undirected_w[i])
    #   -----------------------------
    # Kruskal's Minimum expanding tree
    print('------------------------------------------------')
    print('Running Kruskal\'s algorithm')
    mst_edges = g.kruskal_mst()
    print(mst_edges)
    #   -----------------------------
    # Prim's Minimum expanding tree
    print('------------------------------------------------')
    print('Running Prim\'s algorithm')
    mst_edges = g.prim_mst()
    print(mst_edges)
    #   -----------------------------
    # Floyd-Warshall's all pairs shortest path algorithm
    print('------------------------------------------------')
    print('Running Floyd-Warshall\'s algorithm')
    distance_mat, path_mat = g.floyd_warshall(w)
    # check the path from node 0 to node 4
    path = g.fl_reader(0, 4, path_mat)
    print(path)
    #   -----------------------------
    # Ford_Fulkerson maximum flow problem
    print('------------------------------------------------')
    print('Ford_Fulkerson maximum flow problem')
    print(g.ford_fulkerson(0, num_nodes-1, w))
    #   -----------------------------
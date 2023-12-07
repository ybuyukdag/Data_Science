# Source: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
# Python program for Kruskal's algorithm to find 
# Minimum Spanning Tree of a given connected, 
# undirected and weighted graph 
# This code is contributed by Neelam Yadav 
# Improved by James Graça-Jones 
# I took this python code blocks and improved it with networkx library to show plot of the graph that we took from the result of KruskalMST Algorithm

import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    # Function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i 
	# (truly uses path compression technique)
    def find(self, parent, i):
        if parent[i] != i:

            # Reassignment of node's parent 
			# to root node as 
			# path compression requires 
			
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    # A function that does union of two sets of x and y 
	# (uses union by rank) 
    def union(self, parent, rank, x, y):

        # Attach smaller rank tree under root of 
		# high rank tree (Union by Rank) 
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1

    # The main function to construct MST 
	# using Kruskal's algorithm 
    def KruskalMST(self):

        # This will store the resultant MST 
        self.result = []

        # An index variable, used for sorted edges 
        i = 0

        # An index variable, used for result[] 
        e = 0

        # Sort all the edges in 
		# non-decreasing order of their 
		# weight 
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements 
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is less than to V-1 / to prevent the graph to make a circle
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                self.result.append([u, v, w])
                self.union(parent, rank, x, y)

        minimumCost = 0
        print("Edges in the constructed MST")
        for u, v, weight in self.result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree", minimumCost)

    def visualizeMST(self):
        G = nx.Graph()

        #convert nodes with alphabet label
        alphabet = "abcdefghıjklmnopqrstuvwxwz"

        node_labels = {}
        for i in range(self.V):
            node_labels[i] = alphabet[i]
            G.add_node(i, label = alphabet[i])

        for u, v, w in self.result:
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G)  # Weighted graph layout from networkx lib

        nx.draw(G, pos, with_labels=False, node_size=700, node_color="skyblue", font_color="black")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")


        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        plt.title("Minimum Spanning Tree")
        plt.show()

# Driver code
# if __name__ == '__main__':
#     g = Graph(8)
#     g.addEdge(0, 1, 2)
#     g.addEdge(0, 6, 4)
#     g.addEdge(1, 2, 7)
#     g.addEdge(1, 7, 2)
#     g.addEdge(2, 3, 3)
#     g.addEdge(2, 7, 12)
#     g.addEdge(3, 7, 5)
#     g.addEdge(3, 4, 6)
#     g.addEdge(3, 6, 5)
#     g.addEdge(4, 5, 7)
#     g.addEdge(5, 6, 3)
#     g.addEdge(6, 7, 3)
    

#     g.KruskalMST()
#     g.visualizeMST()


if __name__ == '__main__':
    g = Graph(6)
    g.addEdge(0, 1, 3)
    g.addEdge(0, 4, 6)
    g.addEdge(0, 5, 5)
    g.addEdge(1, 2, 1)
    g.addEdge(1, 5, 4)
    g.addEdge(2, 3, 6)
    g.addEdge(2, 5, 4)
    g.addEdge(3, 5, 5)
    g.addEdge(3, 4, 8)
    g.addEdge(4, 5, 2)


    g.KruskalMST()
    g.visualizeMST()

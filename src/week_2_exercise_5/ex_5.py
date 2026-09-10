import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from week_2_exercise_5.my_reach import my_reach
A = np.zeros((5, 5))
A[0, 1] = 1
A[1, 2] = 1
A[1, 4] = 1
A[2, 3] = 1
A = 100 * np.eye(5) + A + A.T

edges = []

for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[1]):
        if A[i, j] != 0:
            edges.append((i + 1, j + 1))

print(A)
# R, visited = my_reach(A, 2, [1])
# print("Reachable nodes from node 1:", R)
# G = nx.Graph()
# G.add_edges_from(edges)
# nx.draw(G, with_labels=True, font_weight='bold', node_color='lightblue', node_size=500)
# plt.show()
# print(A)
# print(edges)
# print(G.edges())
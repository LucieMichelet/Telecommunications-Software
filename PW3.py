# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 19:54:36 2022

@author: Lucie
"""

#%%------------------------------------Practical Work N°2 - Task N°1-------------------------------------------
#%%

from queue import PriorityQueue

#%%----------------------------------------Dijkstra Algorithm---------------------------------------------

class Graph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
        self.visited = []
        
    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight
        


def dijkstra(graph, start_vertex):
    D = {v:float('inf') for v in range(graph.v)}
    D[start_vertex] = 0

    pq = PriorityQueue()
    pq.put((0, start_vertex))

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        graph.visited.append(current_vertex)

        for neighbor in range(graph.v):
            if graph.edges[current_vertex][neighbor] != -1:
                distance = graph.edges[current_vertex][neighbor]
                if neighbor not in graph.visited:
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost < old_cost:
                        pq.put((new_cost, neighbor))
                        D[neighbor] = new_cost
    return D


#%%---------------------------------------Run the chosen figure---------------------------------
#%%Figure 1

Fig1 = [('u','v',2),('u','w',5),('u','x',1),('v','w',3),('v','x',2),
        ('w','z',5),('w','y',1),('w','x',3),('x','y',1),('z','y',1)]

l = ['u','v','w','x','y','z']

g = Graph(len(l))
g.add_edge(0, 1, 2)
g.add_edge(0, 2, 5)
g.add_edge(0, 3, 1)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 2)
g.add_edge(2, 5, 5)
g.add_edge(2, 3, 3)
g.add_edge(3, 4, 1)
g.add_edge(4, 5, 1)

    
#%% Figure 2 

Fig2 = [('A','B',3),('A','F',6),('A','E',1),('B','E',1),
        ('B','C',4),('C','D',9),('D','E',1),('E','F',2)]

l = ['A','B','C','D','E','F']

g = Graph(len(l))
g.add_edge(0, 1, 3)
g.add_edge(0, 4, 1)
g.add_edge(0, 5, 6)
g.add_edge(1, 2, 4)
g.add_edge(2, 3, 9)
g.add_edge(3, 4, 1)
g.add_edge(4, 5, 2)


#%%Figure 3

Fig3 = [('u','v',3),('u','w',2),('v','y',2),('v','x',1),
        ('w','s',4),('t','s',3),('x','z',3),('x','t',5),('y','z',1)]

l = ['u','v','w','x','y','z','t','s']

g = Graph(len(l))
g.add_edge(0, 1, 3)
g.add_edge(0, 2, 2)
g.add_edge(1, 3, 1)
g.add_edge(1, 4, 2)
g.add_edge(1, 5, 1)
g.add_edge(2, 7, 4)
g.add_edge(3, 5, 4)
g.add_edge(3, 6, 3)
g.add_edge(6, 7, 3)


#%%Figure 4

Fig4 = [('A','B',2),('A','C',3),('B','C',1),('B','D',1),
        ('B','E',4),('C','F',5),('F','E',1),('F','G',1)]

l = ['A','B','C','D','E','F','G']

g = Graph(len(l))
g.add_edge(0, 1, 2)
g.add_edge(0, 2, 3)
g.add_edge(1, 2, 1)
g.add_edge(1, 3, 1)
g.add_edge(1, 4, 4)
g.add_edge(2, 5, 5)
g.add_edge(4, 5, 1)
g.add_edge(5, 6, 1)



#%%Figure 5

Fig5 = [('A','B',5),('A','C',10),('B','C',3),('B','D',11),('C','D',2)]

l = ['A','B','C','D']

g = Graph(len(l))
g.add_edge(0, 1, 5)
g.add_edge(0, 2, 10)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 11)
g.add_edge(2, 3, 2)



#%%Figure exam 2

Fig3 = [('u','v',3),('u','w',3),('u','t',2),('v','w',4),('v','x',3),('v','y',8),('v','t',4),
        ('w','x',6),('x','y',6),('x','z',8),('y','z',12),('y','t',7)]

l = ['u','v','w','x','y','z','t']

g = Graph(len(l))
g.add_edge(0, 1, 3)
g.add_edge(0, 2, 2)
g.add_edge(0, 6, 2)
g.add_edge(1, 2, 4)
g.add_edge(1, 3, 3)
g.add_edge(1, 4, 8)
g.add_edge(1, 6, 4)
g.add_edge(2, 3, 6)
g.add_edge(3, 4, 6)
g.add_edge(3, 5, 8)
g.add_edge(4, 5, 12)
g.add_edge(4, 6, 7)


#%%------------------------------------------Run the algorithm---------------------------------------------


start = 5
D = dijkstra(g, start)


for vertex in range(len(D)):
    print("Distance from ",l[start], " to ", l[vertex], "is", D[vertex])
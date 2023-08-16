# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 19:00:02 2023

@author: Lucie
"""

# A utility function to search a given key in BST
import numpy as np

#%% Biblio
def search(root, key):

    # Base Cases: root is null or key is present at root
    if root is None or root.val == key:
        return root

    # Key is greater than root's key
    if root.val < key:
        return search(root.right, key)

    # Key is smaller than root's key
    return search(root.left, key)

# insert operation in binary search tree

# A utility class that represents
# an individual node in a BST


class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# A utility function to insert
# a new node with the given key


def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if root.val == key:
            return root
        elif root.val < key:
            root.right = insert(root.right, key)
        else:
            root.left = insert(root.left, key)
    return root

# A utility function to do inorder tree traversal


def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)



#%% We import the data

Ebrut = np.genfromtxt("SDN_traffic.csv", dtype=str, delimiter=',')
Elabelscolonne = Ebrut[0, :-1]  # columns
Elabelsligne = Ebrut[1:, -1]  # category
E = Ebrut[1:, :-1]

#%% We get all the categories
catstr = []

for i in range(len(Elabelsligne)-1):                                #get a list of the categories
    x = Elabelsligne[i]
    xi = Elabelsligne[i+1]
    if x != xi and x not in catstr:
        catstr.append(x)
catstr = list(map(lambda x: str(x), catstr))
        
#%%

for c, x in enumerate(np.unique(Elabelsligne)):
    Elabelsligne[Elabelsligne == x] = c                             #convert category into numbers
Elabelsligne = list(map(lambda x: int(x), Elabelsligne))            #convert into int

cat = []

for i in range(len(Elabelsligne)-1):                                #get a list of the categories
    x = Elabelsligne[i]
    xi = Elabelsligne[i+1]
    if x != xi and x not in cat:
        cat.append(x)

#%% We construct and organize the node

n = Node(cat[0])
for i in range(len(cat)):
    n = insert(n,cat[i])
    
inorder(n)

#%%Then we have the order of categories chosen by the BST algorithm

orderdedlabels = []

for i in cat:
    orderdedlabels.append(catstr[i])






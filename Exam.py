# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:47:12 2023

@author: Lucie
"""

#%% Exercise 1

alpha = 0.9
sigma = 1/8
teta = 4
phi = 1

sample = 1
est = 4
dev = 0.5
Time = phi*est + teta*dev
i =0 

while Time > 4 :
    dif = sample - est
    est = alpha*est + (1-alpha)*sample
    dev = dev + sigma*(abs(dif)-dev)
    Time = phi*est + teta*dev
    i +=1
    
#%%Exercise 4.1

stack = []

stack.append(5)
stack.append(3)
print(stack)
stack.pop()
print(stack)
stack.append(2)
stack.append(8)
print(stack)
stack.pop()
stack.pop()
print(stack)
stack.append(9)
stack.append(1)
print(stack)
stack.pop()
print(stack)
stack.append(7)
stack.append(6)
print(stack)
stack.pop()
stack.pop()
print(stack)
stack.append(4)
print(stack)
stack.pop()
stack.pop()
print(stack)

#%% Exercise 4.3

S = [1,2,3,4,5]
T = []

for i in range(len(S)):
    T.append(S[-1])
    S.pop()
    
print(T)

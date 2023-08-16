# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:07:45 2022

@author: Lucie
"""

import numpy as np
import biblio as b

#%% Importation of the excel file

Ebrut = np.genfromtxt("SDN_traffic.csv", dtype = str, delimiter=',')
Elabelscolonne = Ebrut[0,:-1]                                                   #columns
Elabelsligne = Ebrut[1:,-1]                                                     #category
E = Ebrut[1:,:-1] 

for color, x in enumerate(np.unique(Elabelsligne)):
    Elabelsligne[Elabelsligne == x] = color
Elabelsligne = list(map(lambda x: int(x), Elabelsligne))

#%%

print(b.ACP2D(E,Elabelsligne,Elabelscolonne))


#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from random import choice
import matplotlib.pyplot as plt

from Basic_functions import *


##########################################################
#                                                        #
#   Data: compute empirical pairwise probabilities       #
#                                                        #
##########################################################


cars_number = 20 # 10 or 20

if cars_number == 20:
    df = pd.read_csv('data/cars/exp2-prefs/prefs2.csv')
    K0, K1 = 3, 10
elif cars_number == 10:
    df = pd.read_csv('data/cars/exp1-prefs/prefs1.csv')
    K0, K1 = 3, 8


items=list(np.unique(df[' Item1 ID']))+list(np.unique(df[' Item2 ID']))
items=np.unique(items)

n_items=len(items)
all_pairs=[(i,j) for (i,j) in combinations(items,2)]
index_pairs={(i,j):u for u,(i,j) in enumerate(all_pairs)}

data=[(u,v) for (u,v) in zip(df[' Item1 ID'], df[' Item2 ID'])]
count=Counter(data)

proba_pairs=compute_proba_pairs(count,index_pairs)
copeland_emp=Copeland(proba_pairs)


##########################################################
#                                                        #
#   Plot dimension /distortion for different K/lambda    #
#                                                        #
##########################################################



nb_buckets=range(K0,K1+1)
all_K_distortions=[]
all_K_penalties=[]

for K in nb_buckets:
    distortions, penalties=compute_distortions_penalties(K,n_items, copeland_emp,proba_pairs)
    all_K_distortions.append(distortions)
    all_K_penalties.append(penalties)

distortions_list, penalties_list = [], []
for K in nb_buckets:
    i=nb_buckets.index(K)
    distortions=all_K_distortions[i]
    penalties=all_K_penalties[i]
    if cars_number==20:
        if (K>K0) and (len(distortions)>len(distortions_list[0])):
            distortions=distortions[::len(distortions_list[0])]
            penalties=penalties[::len(distortions_list[0])]
    distortions_list.append(distortions)
    penalties_list.append(penalties)


plot_expe(penalties_list, distortions_list, n_items, nb_buckets, '%d cars dataset' % cars_number, 'plots/cars_dataset_%d_scaled.pdf' % cars_number)


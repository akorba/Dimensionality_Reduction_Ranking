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


f = open('data/sushi3a.5000.10.order','r')
unique_users=[]
n_items=10
items=range(0,n_items)


sushi_orders=[]
for line in f:
    preferences=[]
    l = map(int,line.split(' '))
    sushi_orders.append(tuple(l[2:]))


n_items=len(items)
all_pairs=[(i,j) for (i,j) in combinations(items,2)]
index_pairs={(i,j):u for u,(i,j) in enumerate(all_pairs)}


data=[]
for l in sushi_orders:
    preferences=[]
    for (i,j) in combinations(items,2):
        data.append(induced_subword(l,(i,j)))

count=Counter(data)
proba_pairs=compute_proba_pairs(count, index_pairs)
copeland_emp=Copeland(proba_pairs)


##########################################################
#                                                        #
#   Plot dimension /distortion for different K/lambda    #
#                                                        #
##########################################################

K0,K1=3,8
nb_buckets=range(K0,K1+1)

all_K_distortions=[]
all_K_penalties=[]

for K in nb_buckets:
    distortions, penalties, buckets=compute_distortions_penalties(K,n_items,copeland_emp,proba_pairs,sushi_orders)
    all_K_distortions.append(distortions)
    all_K_penalties.append(penalties)

distortions_list, penalties_list = [], []

for K in nb_buckets:
    i=nb_buckets.index(K)
    distortions=all_K_distortions[i]
    penalties=all_K_penalties[i]
    distortions_list.append(distortions)
    penalties_list.append(penalties)

plot_expe(penalties_list, distortions_list, n_items, nb_buckets, '10 sushi dataset', 'plots/sushi_dataset_scaled.pdf')

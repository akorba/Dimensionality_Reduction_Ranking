#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import random
import numpy as np
from Basic_functions import *

###### compare true bucket distribution and uniform distribution for 10 and 20 items.

n_items=20

if n_items==10:
    bucket_order= [[0,1,2,3],[4,5], [6,7,8], [9,10]]
    #max_y_axis_uniform=25
    

if n_items==20:
    bucket_order= [[0,1,2, 3,4],[5,6,7], [8, 9], [10,11,12,13], [14,15,16,17],[18,19]]
    #max_y_axis_uniform=90


items=range(0,n_items)
all_pairs=[(i,j) for (i,j) in combinations(items,2)]
index_pairs={(i,j):u for u,(i,j) in enumerate(all_pairs)}


#####################################
#                                   #
#  True bucket distribution         #
#                                   #
#####################################


# define the true bucket distribution


p_c={}
for (a,b) in combinations(items,2):
    index_a= [(i, bucket.index(a)) for i, bucket in enumerate(bucket_order) if a in bucket][0]
    index_b= [(i, bucket.index(b)) for i, bucket in enumerate(bucket_order) if b in bucket][0]
    if index_a[0]<index_b[0]:
        p_c[(a,b)]= 1
    elif index_a[0]>index_b[0]:
        p_c[(a,b)]= 0
    else:
        p_c[(a,b)]= random.uniform(0, 1)

copeland_true=Copeland(p_c)

nb_buckets=range(3,9)
all_K_distortions=[]
all_K_penalties=[]
all_K_buckets=[]


for K in nb_buckets:
    distortions1, penalties1, buckets1=compute_distortions_penalties(K,n_items, copeland_true,p_c, True)
    if n_items==20: # if n_items=20 we have to subsample the number of points
        if (K>3) and (len(distortions1)>len(all_K_distortions[0])):
            distortions1=distortions1[::len(all_K_distortions[0])]
            penalties1=penalties1[::len( all_K_distortions[0])]
    all_K_distortions.append(distortions1)
    all_K_penalties.append(penalties1)

plot_expe(all_K_penalties, all_K_distortions, n_items, nb_buckets, 'true bucket distribution', 'plots/true_bucket_distribution_%d_items.pdf' % n_items)


#####################################
#                                   #
#      Uniform distribution         #
#                                   #
#####################################

p_c={}
for (a,b) in combinations(items,2):
    p_c[(a,b)]= 0.5

copeland_true=Copeland(p_c)

nb_buckets=range(3,9)
all_K_distortions=[]
all_K_penalties=[]
all_K_buckets=[]

for K in nb_buckets:
    distortions1, penalties1, buckets1=compute_distortions_penalties(K,n_items, copeland_true,p_c, True)
    if n_items==20: # if n_items=20 we have to subsample the number of points
        if (K>3) and (len(distortions1)>len(all_K_distortions[0])):
            distortions1=distortions1[::len(all_K_distortions[0])]
            penalties1=penalties1[::len( all_K_distortions[0])]
    all_K_distortions.append(distortions1)
    all_K_penalties.append(penalties1)


plot_expe(all_K_penalties, all_K_distortions, n_items, nb_buckets, 'uniform distribution', 'plots/uniform_distribution_%d_items.pdf' % n_items)
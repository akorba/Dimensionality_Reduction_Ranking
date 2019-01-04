#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import Counter

from Basic_functions import *


###### what happens if we add some noise to a bucket distribution?

n_items=6
items=range(0,n_items)
all_pairs=[(i,j) for (i,j) in combinations(items,2)]
index_pairs={(i,j):u for u,(i,j) in enumerate(all_pairs)}


########################################
#                                      #
#     True bucket distribution         #
#                                      #
########################################


# define the true bucket distribution

bucket_order= [[1,5], [2,3,4], [0]]

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

nb_buckets=range(2,4)
all_K_distortions=[]
all_K_penalties=[]
all_K_buckets=[]


for K in nb_buckets:
    distortions1, penalties1, buckets1=compute_distortions_penalties(K,n_items, copeland_true,p_c, True)
    all_K_buckets.append(buckets1)
    all_K_distortions.append(distortions1)
    all_K_penalties.append(penalties1)
    best_distortion=np.min(distortions1)
    best_bucket=buckets1[np.argmin(distortions1)]
    print 'best bucket of size '+str(K)+' is '+str(best_bucket)+' with distortion ' +str(best_distortion)

plot_expe(all_K_penalties, all_K_distortions, n_items, nb_buckets, 'true bucket distribution', 'plots/true_bucket_distribution.pdf')


########################################
#                                      #
#     Noisy distributions              #
#                                      #
########################################

bucket_distribution = dict.fromkeys(permutations(range(0,n_items)))

permutations_of_buckets=list([list(permutations(bucket)) for bucket in bucket_order])
sigmas_bucket=list(itertools.product(*permutations_of_buckets))
sigmas_bucket=[ tuple([element for tupl in tupleoftuples for element in tupl]) for tupleoftuples in sigmas_bucket]

norm_constant2=len(sigmas_bucket)

for sigma in bucket_distribution.keys():
    if sigma in sigmas_bucket:
        bucket_distribution[sigma]=1/(norm_constant2*1.0)
    else:
        bucket_distribution[sigma]=0

# test: you can check that the marginals of bucket_distribution are the same as p_c
#p_c2=compute_true_proba_pairs(bucket_distribution, all_pairs)

############### ADD 10% noise

dataset= [sample_from(bucket_distribution) for i in range(2000)]

# swap two items
for s in range(len(dataset)):
    dataset[s]=list(dataset[s])
    if random.uniform(0,1)<0.1:
        (i,j)=random.sample(items,2)
        dataset[s][i], dataset[s][j]=dataset[s][j], dataset[s][i]
    dataset[s]=tuple(dataset[s])

data=[]
for l in dataset:
    preferences=[]
    for (i,j) in combinations(items,2):
        data.append(induced_subword(l,(i,j)))

count=Counter(data)
proba_pairs=compute_proba_pairs(count, index_pairs)
copeland_emp=Copeland(proba_pairs)


nb_buckets=range(2,4)
all_K_distortions=[]
all_K_penalties=[]
all_K_buckets=[]


for K in nb_buckets:
    distortions1, penalties1, buckets1=compute_distortions_penalties(K,n_items, copeland_emp,proba_pairs, True)
    all_K_buckets.append(buckets1)
    all_K_distortions.append(distortions1)
    all_K_penalties.append(penalties1)
    best_distortion=np.min(distortions1)
    best_bucket=buckets1[np.argmin(distortions1)]
    print 'best bucket of size '+str(K)+' is '+str(best_bucket)+' with distortion ' +str(best_distortion)

plot_expe(all_K_penalties, all_K_distortions, n_items, nb_buckets, '10% noisy bucket distribution', 'plots/10_noisy_bucket_distribution.pdf')


############### ADD 20% noise


dataset= [sample_from(bucket_distribution) for i in range(2000)]

for s in range(len(dataset)):
    dataset[s]=list(dataset[s])
    if random.uniform(0,1)<0.2:
        (i,j)=random.sample(items,2)
        dataset[s][i], dataset[s][j]=dataset[s][j], dataset[s][i]
    dataset[s]=tuple(dataset[s])

data=[]
for l in dataset:
    preferences=[]
    for (i,j) in combinations(items,2):
        data.append(induced_subword(l,(i,j)))

count=Counter(data)
proba_pairs=compute_proba_pairs(count, index_pairs)
copeland_emp=Copeland(proba_pairs)


nb_buckets=range(2,4)
all_K_distortions=[]
all_K_penalties=[]
all_K_buckets=[]


for K in nb_buckets:
    distortions1, penalties1, buckets1=compute_distortions_penalties(K,n_items, copeland_emp,proba_pairs, True)
    all_K_buckets.append(buckets1)
    all_K_distortions.append(distortions1)
    all_K_penalties.append(penalties1)
    best_distortion=np.min(distortions1)
    best_bucket=buckets1[np.argmin(distortions1)]
    print 'best bucket of size '+str(K)+' is '+str(best_bucket)+' with distortion ' +str(best_distortion)


plot_expe(all_K_penalties, all_K_distortions, n_items, nb_buckets, '20% noisy bucket distribution','plots/20_noisy_bucket_distribution.pdf')


############### ADD 50% noise


dataset= [sample_from(bucket_distribution) for i in range(2000)]

for s in range(len(dataset)):
    dataset[s]=list(dataset[s])
    if random.uniform(0,1)<0.5:
        (i,j)=random.sample(items,2)
        dataset[s][i], dataset[s][j]=dataset[s][j], dataset[s][i]
    dataset[s]=tuple(dataset[s])

data=[]
for l in dataset:
    preferences=[]
    for (i,j) in combinations(items,2):
        data.append(induced_subword(l,(i,j)))

count=Counter(data)
proba_pairs=compute_proba_pairs(count, index_pairs)
copeland_emp=Copeland(proba_pairs)


nb_buckets=range(2,4)
all_K_distortions=[]
all_K_penalties=[]
all_K_buckets=[]


for K in nb_buckets:
    distortions1, penalties1, buckets1=compute_distortions_penalties(K,n_items, copeland_emp,proba_pairs, True)
    all_K_buckets.append(buckets1)
    all_K_distortions.append(distortions1)
    all_K_penalties.append(penalties1)
    best_distortion=np.min(distortions1)
    best_bucket=buckets1[np.argmin(distortions1)]
    print 'best bucket of size '+str(K)+' is '+str(best_bucket)+' with distortion ' +str(best_distortion)



plot_expe(all_K_penalties, all_K_distortions, n_items, nb_buckets, '50% noisy bucket distribution','plots/50_noisy_bucket_distribution.pdf')

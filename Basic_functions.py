#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from random import *
import random
import numpy as np
import math
from itertools import combinations, product
import matplotlib.pyplot as plt
import itertools
import copy



# sample from distribution (dict) d
def sample_from(d):
    r = uniform(0, sum(d.itervalues()))
    s = 0.0
    for k, w in d.iteritems():
        s += w
        if r < s: return k
    return k


################################################
#                                              #
#   compute pairwise probabilities             #
#                                              #
################################################

# compute empirical pairwise probabilities
def compute_proba_pairs(data_pairs, index_pairs):
    proba_pairs=dict()
    all_pairs=index_pairs.keys()
    for (i,j) in data_pairs.keys():
        if i<j:
            index_pair=index_pairs[(i,j)]
            p_ij=data_pairs[(i,j)]
            p_ji=data_pairs[(j,i)]
            norm=p_ij+p_ji
            p_ij=p_ij/(1.0*norm)
            proba_pairs[(i,j)]=p_ij
    for (i,j) in all_pairs:
        if (i,j) not in proba_pairs:
            proba_pairs[(i,j)]=0.5
    return proba_pairs

# compute subranking of ranking w on subset A
def induced_subword(w,A):
    t = ()
    for i in w:
        if i in A:
            t = t + (i,)
    return t

# compute marginal of distribution h on subset A
def Restrict(h, A):
    h_=dict.fromkeys(itertools.permutations(A),0)
    for sigma in h.keys():
        sigma_=induced_subword(sigma,A)
        h_[sigma_]+=h[sigma]
    return h_

# compute pairwise marginals of (true) distribution P
def compute_true_proba_pairs(P, all_pairs):
    proba_pairs=dict()
    for pair in all_pairs:
        m=all_pairs.index(pair)
        i,j=pair
        p_pair=Restrict(P,(i,j))
        p_ij= p_pair[(i,j)]
        proba_pairs[pair]=p_ij
    return proba_pairs



def Copeland(pairs):
    A=set(np.unique(pairs.keys()))
    Copeland = dict.fromkeys(A,0)
    for i in A:
        Abis=list(copy.copy(A))
        Abis.remove(i)
        for j in Abis:
            if i<j:
                p_ij=pairs[(i,j)]
            else:
                p_ij=1-pairs[(j,i)]
            if p_ij < 0.5:
                Copeland[i] += 1
    for i in A:
        Copeland[i]= np.around(Copeland[i], decimals=10)
    pi = ()
    
    for key,value in sorted(Copeland.iteritems(), key=lambda (k,v):(v,k)):   
        pi = pi + (key,)
    return pi 


################################################
#                                              #
#          functions for buckets               #
#                                              #
################################################

def tree_to_bucket(tree):
    visited = []
    bucket=[]

    def dfs(tree):
        #global visited
        index=tree['index']
        if index not in visited:
            visited.append(index)
            for n in [tree['left'],tree['right']]:
                if type(n)==dict:
                    dfs(n)
                else:
                    bucket.append(list(n))

    dfs(tree)
    return bucket


def compute_bucket_pairs(bucket):
    K=len(bucket)
    pairs=[]
    for (k,l) in combinations(range(K),2):
        b_k=bucket[k]
        b_l=bucket[l]
        cross=[pair for pair in product(*[b_k,b_l])]
        pairs+=cross
    return pairs


# computes all buckets shapes lambda for buckets of size K
def compute_all_possible_buckets_indexes(K, n_items):
    all_bucket_indexes=[]
    possible_splits=list(set(range(1,n_items)))
    for splits in combinations(possible_splits, K-1):
        bucket_indexes=[0]+list(splits)+[n_items]
        all_bucket_indexes.append(bucket_indexes)
    return all_bucket_indexes

# computes all buckets of (length K) and shape lambda/sizes
def compute_all_possible_buckets(sizes):
    n_items=np.sum(sizes)
    K=len(sizes)
    all_buckets=[]
    lambd=sizes
    permutations=list(itertools.permutations(range(0,n_items)))
    for sigma in permutations:
        index_begin=0
        index_end=lambd[0]
        bucket=[sigma[index_begin:index_end]]
        for k in range(1,K):
            index_begin=index_end
            index_end=index_end+lambd[k]
            bucket+=[sigma[index_begin:index_end]]
        all_buckets.append(bucket)
    return all_buckets



################################################
#                                              #
#               Distortion                     #
#                                              #
################################################

# compute distortion
# index pairs = (i,j) \in [1,i1]x [i1+1,n] where i1 is the split
def compute_lambda(proba_pairs, index_pairs):
    lambd=0
    size=len(index_pairs)
    if size==0:
        lambd=1e6
    else:
        for m in index_pairs:
            i,j=m[0],m[1]
            if i<j:
                p_ji=1- proba_pairs[m]
            else:
                p_ji=proba_pairs[m[::-1]]
            lambd += p_ji
    return lambd


################################################
#                                              #
#                Penalties                     #
#                                              #
################################################

# compute penalty (dimension)
def compute_penalty1(sizes):
    factorials=[math.factorial(size) for size in sizes]
    pen=np.prod(factorials)-1
    return pen



################################################
#                                              #
#                  Plots                       #
#                                              #
################################################
    
def compute_distortions_penalties(K, n_items, order,proba_pairs, return_buckets=False): 
    all_bucket_indexes=compute_all_possible_buckets_indexes(K,n_items)
    distortions=[]
    penalties=[]
    buckets=[]
    for i in range(len(all_bucket_indexes)):
        bucket_indexes=all_bucket_indexes[i]
        bucket=[list(order[bucket_indexes[i]:bucket_indexes[i+1]]) for i in range(len(bucket_indexes)-1)]
        buckets.append(bucket)
        index_pairs= compute_bucket_pairs(bucket)
        distortion=compute_lambda(proba_pairs, index_pairs)
        #if distortion ==0:
        #    print 'distortion null for '+str(bucket)
        distortions.append(distortion)
        sizes=[len(bucket[i]) for i in range(len(bucket))]
        penalty=compute_penalty1(sizes)
        penalties.append(penalty)
    if return_buckets==False:
        return distortions, penalties
    else:
        return distortions, penalties, buckets

def plot_expe(penalties_list, distortions_list, n_items, nb_buckets,title='', save_name=''):
    # horizontal scale depending on the number of items
    if n_items==10:  
        max_y_axis=20 # change to 25 for test_donnees_simulees2
    elif n_items==6:
        max_y_axis=2.5
    else:
        max_y_axis=90 # change to 90 for test_donnees_simulees2
    f, ax = plt.subplots(figsize=(2.5, 2))
    iter_list = zip(distortions_list, penalties_list, nb_buckets)
    for distortions, penalties, K in iter_list:
        artist1 = ax.set_xlabel('dimension')
        artist2 = ax.set_ylabel('distortion')
        ax.set_ylim([0,max_y_axis])
        ax.set_title(title)
        ax.scatter(penalties, distortions, label='%d' % K, s=5)
        ax.set_xscale('log')
    ax.legend(title='K',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(False)
    plt.savefig(save_name,
                bbox_extra_artists=(artist1, artist2),
                bbox_inches='tight')
    plt.show()

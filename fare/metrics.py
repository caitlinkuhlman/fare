"""Metrics to assess the fairness of a ranker

    References
    ----------
    Caitlin Kuhlman, MaryAnn VanValkenburg, Elke Rundensteiner. 
    "FARE: Diagnostics for Fair Ranking using Pairwise Error Metrics" 
    in the proceedings of the Web Conference (WWW 2019)
"""

# Authors: Caitlin Kuhlman <cakuhlman@wpi.edu>
# License: BSD 3 clause

import numpy as np

__ALL__ = [
    "rank_parity",
    "rank_equality",
    "rank_calibration"
]


def _pairs(n):
    return n*(n-1)/2.



#calibration
def _merge_cal(h1,h2,g):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j < len(h2):
        #pairs that are correctly ordered
        if h1[i][1] < h2[j][1]:
            arr.append(h1[i])
        else:
            while(j<len(h2) and h1[i][1] > h2[j][1]):
                # count inverted pairs containing group
                if h2[j][2] == g:
                    count += len(h1[i:])
                else:
                    count += np.bincount(np.array(h1, dtype=int)[i:,2], minlength=2)[g]
                arr.append(h2[j])
                j+=1
            arr.append(h1[i])
        i+=1
#     add any remaining elements
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count

    
#equality
def _merge_eq(h1,h2,g):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j< len(h2):
        #pairs that are correctly ordered
        if h1[i][1] < h2[j][1]:
            arr.append(h1[i])
        else:
            while(j<len(h2) and h1[i][1] > h2[j][1]):
                if(h2[j][2] != g):
                # count inverted pairs favoring group
                    count += np.bincount(np.array(h1, dtype=int)[i:,2], minlength=2)[g]
                arr.append(h2[j])
                j+=1
            arr.append(h1[i])
        i+=1
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count

# parity
def _merge_parity(h1,h2,g):
    count1 = 0
    count2 = 0
    i=0
    while i < len(h1): 
        if h1[i]== g:
            count1 +=1
        i+=1
    i=0
    while i < len(h2):
        if h2[i] != g:
            count2 +=1  
        i+=1
    count = count1*count2   
    return np.concatenate([h1,h2]), count


#compute FARE error metrics using adaptation of mergesort to perform pair counting.	
def _count_inversions(data, s, e, merge_fnc, g):
    if s == e: #base case
        return [data[s]], 0
    else:
        m = s + int((e-s)/2)
        h1,c1 = _count_inversions(data, s, m, merge_fnc, g)
        h2,c2 = _count_inversions(data, m+1, e, merge_fnc, g)
        merged, c = merge_fnc(h1,h2,g)
        return merged, (c1+c2+c)


def rank_equality(y_true, y_pred, groups):
    """Compute the rank equality error between two rankings.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples)
        Estimated target values.

    groups : array-like of shape = (n_samples)
        Binary integer array with group labels for each sample. 

    Returns
    ----------
    error0 : float
        The rank parity error for group 0.

    error1 : float
        The rank parity error for group 1.

    Examples
    --------
    >>> y_true = [1,2,3,4]
    >>> y_pred = [1,3,4,2]
    >>> groups =[0,1,0,1]
    >>> rank_equality(y_true,y_pred,groups)
    (0.0, 0.25)
    """
    #sort instances by y_pred
    r = np.transpose([y_true,y_pred,groups])
    r = r[r[:,0].argsort()]
    #count the items in each group for narmalization
    len_groups = np.bincount(np.array(groups, dtype=int), minlength=2)
    p = len_groups[0]*len_groups[1]
    e0 = 0 if p == 0 else _count_inversions(r, 0, len(r)-1, _merge_eq, 0)[1] / p
    e1 = 0 if p == 0 else _count_inversions(r, 0, len(r)-1, _merge_eq, 1)[1] / p
    return e0, e1


def rank_calibration(y_true, y_pred, groups):
    """Compute the rank calibration error between two rankings.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples)
        Estimated target values.

    groups : array-like of shape = (n_samples)
        Binary integer array with group labels for each sample. 

    Returns
    -------
    error0 : float
        The rank parity error for group 0.

    error1 : float
        The rank parity error for group 1.

    Examples
    --------
    >>> y_true = [1,2,3,4]
    >>> y_pred = [1,3,4,2]
    >>> groups =[0,1,0,1]
    >>> rank_calibration(y_true,y_pred,groups)
    (0.20000000000000001, 0.40000000000000002)
    """
    #sort instances by y_pred
    r = np.transpose([y_true,y_pred,groups])
    r = r[r[:,0].argsort()]
    #count the items in each group for normalization
    len_groups = np.bincount(np.array(groups, dtype=int), minlength=2)
    p0 = _pairs(len(r)) - _pairs(len_groups[1])
    p1 = _pairs(len(r)) - _pairs(len_groups[0])
    # count pairs
    e0 = 0 if p0 == 0 else _count_inversions(r, 0, len(r)-1, _merge_cal, 0)[1] / p0
    e1 = 0 if p1 == 0 else _count_inversions(r, 0, len(r)-1, _merge_cal, 1)[1] / p1
    return e0, e1 

def rank_parity(y,groups):
    """Compute the rank parity error for one ranking.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Rank values.
        
    groups : array-like of shape = (n_samples)
        Binary integer array with group labels for each sample.
    
    Returns
    -------
    error0 : float
        The rank parity error for group 0.

    error1 : float
        The rank partiy error for group 1.

    Examples
    --------
    >>> y = [1,3,4,2]
    >>> groups =[0,1,0,1]
    >>> rank_parity(y, groups)
    (0.5,0.5)
    """
    # assume groups vector is in rank order
    #count the items in each group for normalization
    r = np.transpose([y,groups])
    r = r[r[:,0].argsort()]
    g= np.array(r[:,1], dtype=int)
    len_groups = np.bincount(np.array(groups, dtype=int), minlength=2)
    if(len_groups[1] == 0):
        #if there are no group 1 items, group 0 always preferred
        return 1.,0.
    if(len_groups[0] == 0):
        #if there are no group 0 items, group 1 always preferred
        return 0.,1.
    p = len_groups[0]*len_groups[1]
    # if there are no mixed pairs, can't normalize so set both errs = 0
    e0 = _count_inversions(g, 0, len(g)-1, _merge_parity, 0)[1] / p
    e1 = _count_inversions(g, 0, len(g)-1, _merge_parity, 1)[1] / p

    return e0,e1

"""FARE auditing methodology to assess the fairness of a ranker.

    References
    ----------
    Caitlin Kuhlman, MaryAnn VanValkenburg, Elke Rundensteiner. 
    "FARE: Diagnostics for Fair Ranking using Pairwise Error Metrics" 
    in the proceedings of the Web Conference (WWW 2019)
"""

# Authors: Caitlin Kuhlman <cakuhlman@wpi.edu>
# License: BSD 3 clause

import numpy as np
from scipy import stats
from metrics import *

__ALL__ = [
    "audit_parity",
    "audit_equality",
    "audit_calibration"
    "generate_diagnostics"
]



def audit_parity(y, groups, window, step):
    """Generate the error sequences for rank auditing using the rank parity metric. 

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        rank values.

    groups : array-like of shape = (n_samples)
        binary integer array with group labels for each sample. 

    window : int
        number of instances in each bin
        
    step : int
        step size for sliding window
        
    Returns
    -------
    error0: array-like of shape = (n_bins)
        The rank parity error sequence for group 0

    error1: array-like of shape = (n_bins)
        The rank parity error sequence for group 1

    Examples
    --------
    
    """    
    #error sequences
    err0=[]
    err1=[]
    start=0
    end=window
    
    #perform binning
    r = np.transpose([y,groups])
    #sort values by rank value
    r = r[r[:,0].argsort()]

    while end<len(r):
        vals = r[start:end]
        e0,e1 = rank_parity(vals[:,0],vals[:,1])
        err0.append(e0)
        err1.append(e1)
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(r)-window):
        vals = r[len(r)-window:]
        e0,e1 = rank_parity(vals[:,0],vals[:,1])
        err0.append(e0)
        err1.append(e1)
    return err0, err1


def audit_equality(y_true, y_pred, groups, window, step):
    """Generate the error sequences for rank auditing using the rank equality metric. 

    Parameters
    ----------
    y_true : array-like of shape = (n_samples)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples)
        Estimated target values.

    groups : array-like of shape = (n_samples)
        binary integer array with group labels for each sample. 

    window : int
        number of instances in each bin
        
    step : int
        step size for sliding window
        
    Returns
    -------
    error0: array-like of shape = (n_bins)
        The rank equality error sequence for group 0

    error1: array-like of shape = (n_bins)
        The rank equality error sequence for group 1

    Examples
    --------
    
    """     
    #error sequences
    err0=[]
    err1=[]
    start=0
    end=window
    
    #perform binning
    r = np.transpose([y_true,y_pred,groups])
    #sort values by predicted value
    r = r[r[:,1].argsort()]
    while end<len(r):
        vals = r[start:end]
        e0,e1 = rank_equality(vals[:,0],vals[:,1],vals[:,2])
        err0.append(e0)
        err1.append(e1)
        start+=step
        end+=step
    #get end of rank if needed
    if(start > len(r)-window):
        vals = r[len(r)-window:]
        e0,e1 = rank_equality(vals[:,0],vals[:,1],vals[:,2])
        err0.append(e0)
        err1.append(e1)
    return err0, err1


def audit_calibration(y_true, y_pred, groups, window, step):
    """Generate the error sequences for rank auditing using the rank calibration metric. 

    Parameters
    ----------
    y_true : array-like of shape = (n_samples)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples)
        Estimated target values.

    groups : array-like of shape = (n_samples)
        binary integer array with group labels for each sample. 

    window : int
        number of instances in each bin
        
    step : int
        step size for sliding window
        
    Returns
    -------
    error0: array-like of shape = (n_bins)
        The rank calibration error sequence for group 0

    error1: array-like of shape = (n_bins)
        The rank calibration error sequence for group 1

    Examples
    --------
    
    """         
    #error sequences
    err0=[]
    err1=[]
    start=0
    end=window
    
    #perform binning
    r = np.transpose([y_true,y_pred,groups])
    #sort values by predicted value
    r = r[r[:,1].argsort()]
    
    while end<len(r):
        vals = r[start:end]
        e0,e1 = rank_calibration(vals[:,0],vals[:,1],vals[:,2])
        err0.append(e0)
        err1.append(e1)
        start+=step
        end+=step
    #get end of rank if needed
    if(start > len(r)-window):
        vals = r[len(r)-window:]
        e0,e1 = rank_calibration(vals[:,0],vals[:,1],vals[:,2])
        err0.append(e0)
        err1.append(e1)
    return err0, err1


def generate_diagnostics(err0, err1):
    """Generate diagnostic statistics for audit error sequences. 

    Parameters
    ----------
    err0: array-like of shape = (n_bins)
        The rank calibration error sequence for group 0

    err1: array-like of shape = (n_bins)
        The rank calibration error sequence for group 1
        
    Returns
    -------

    trend0 : float
        The trend diagnostic for err0. Computed as the slope of the best fit line.
        
    trend1 : float
        The trend diagnostic for err1. Computed as the slope of the best fit line.
        
    dist : float
        The distance diagnostic for the sequences. Computed as the mean pointwise absolute difference.
        
    Examples
    --------
    
    """  
    diagnostics=[]
    #trends
    r0=[x/len(err0) for x in range(len(err0))]
    r1=[x/len(err1) for x in range(len(err1))]
    diagnostics.append(stats.linregress(r0, y=err0)[0])
    diagnostics.append(stats.linregress(r1, y=err1)[0])
    #correlation
    #errs.append(stats.pearsonr(err0,err1)[0])
    #distance
    diffs = np.abs(np.array(err0) - np.array(err1))
    diagnostics.append(np.mean(diffs))
    #significance
    #errs.append(stats.ttest_ind(err0,err1)[1])
    return diagnostics

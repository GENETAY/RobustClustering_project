# Date : 2020-07-22 11:06

#!/usr/bin/python
# -*- coding: utf-8 -*-
#"""
# Some utils function to evaluate clustering initialisation
#              *************
#              *description*
#              *************
#"""

__author__ = "GENETAY Edouard"
__copyright__ = "Copyright 2020"
__version__ = "5.0"
__maintainer__ = "GENTEAY Edouard"
__email__  = "genetay.edouard@gmail.com"
__status__ = "version beta"

"""
All utils functoin used to test and compare performances of the algorithm of this package.

These functions were used to produce the numerical experiments of the article of Saumard, Saumard, Genetay about K-bMOM
"""


from scipy.stats import mode
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import ot.lp

def flatten_list(list_of_lists):
    res = []
    for l in list_of_lists:
        res += l
    return res


def mapping(target_lbl,clusters):
    '''
    Dictionary which maps the number cluster to the most probable label
    true_label: list of true partition
    cluster   : fitted clusters
    K         : number of clusters
    '''
    mymap = {}
    for i,clu in enumerate(set(clusters)):
        mask = (clusters == clu)
        mymap[i] = mode(target_lbl[mask])[0][0]
    return mymap


def RMSE(centers,centers_pred,mapp):
    '''
    centers: theoritical centers
    centers_hat: fitted centroids
    '''
    k,p = centers.shape
    incr = 0
    for cluster, label in mapp.items():
        incr += sum((centers_pred[int(cluster)] - centers[int(label)])**2)
    return (incr/k)**0.5

def RMSE_ot(centers_ref,centers_test):
    '''
    Function that computes the RMSE between a bunch of theoretical centers 'centers_ref' and
    predicted centers 'centers_test'. These centers must have the same length. The number of
    centers in centers_test and centers_ref can be different, in that case, centers can be
    associated several times. This function is nevertheless not symetric because centers_ref
    will always be taken a as reference to normalize the sum of squares.
    
    :param centers_test: first bunch of centers
    :param centers_ref: second bunch of centers
    
    :type centers_test: ndarray
    :type centers_ref: ndarray
    
    need to import ot.lp
    '''
    if len(centers_test)<len(centers_ref):
        return np.sqrt(ot.dist(x1=centers_test,x2=centers_ref).min(axis=1).sum()/len(centers_ref))
    else:
        return np.sqrt(ot.dist(x1=centers_ref,x2=centers_test).min(axis=1).sum()/len(centers_ref))

def accuracy(true_label,cluster,mapp):
    labels = [mapp[clu] for clu in cluster]
    return accuracy_score(true_label, labels)


def my_inlier_distorsion(data,y_true,y_pred,centers_pred):
    '''
    This function computes the two risks :
    - the one of the predicted partition of the data against the empirical probability measure
    and
    - the one of the true partition of the data against the empirical probability measure
    both are normalized by the percentage of inliers in the data
    '''
    n,p=data.shape
    K_pred,p_pred=centers_pred.shape
    if p_pred!=p: # if different, may mean that transposed centers were given.
        raise Exception("centers must be given in row in a 2d numpy.array")

    # compute the risk value given by the empirical centers against the empirical mean, we correct then the result by the number of data points
    z_distorsion = np.zeros((n,K_pred))#;print(z_true) # z will take data that are inliers for both y_true and y_pred simultaneously
    mask_inliers_pred = np.array([np.arange(n),y_pred])#;print("mask_inliers_pred="+str(mask_inliers_pred))  # mask of the entries of the matrix z to change to 1, one still has to get rid of the -1
    mask_distorsion = np.logical_and(y_true >= 0,y_pred >= 0)#;print("mask_distorsion="+str(mask_distorsion))
    mask_inliers = mask_inliers_pred[:,mask_distorsion].tolist()#;print("mask_inliers="+str(mask_inliers))
    z_distorsion[mask_inliers] = 1#;print("z_distorsion="+str(z_distorsion))
    
    distorsion_inliers = np.sum((data[mask_distorsion]-z_distorsion[mask_distorsion].dot(centers_pred))**2)#;print("distorsion_inliers="+str(distorsion_inliers))
    return(distorsion_inliers,np.sum(mask_distorsion))


def get_metrics(metrics,x,y_true,mu_true,y_pred,mu_pred,outliers_identified_by=-1):
    '''
    this function computes all metrics that are available in known_metrics (see below)
    it outputs the desired metrics in the same order as in the argument 'metrics'
    '''
    from sklearn.metrics import adjusted_rand_score as ari
    known_metrics=["RMSE","ACC","ARI","DISTORSION","n_sample","OT_CENTERS"]
    
    if not any(np.isin(element=known_metrics,test_elements=metrics)):
        raise Exception('all desired metrics are unknown of the function get_metrics')
    if outliers_identified_by!=-1:
        raise exception("outliers must be identified by '-1' in the assignment array y_true and y_pred")
    if type(y_pred[0])!=np.int64 or type(y_true[0])!=np.int64:
        raise Exception("y_pred and y_true must contain integer")
        
    nb_outliers_true = np.sum(y_true==-1)
    # nb_outliers_pred = np.sum(y_pred==-1) # in case it is useful one day, but now it's not used
    
    res=np.zeros(len(metrics))
    for metric in metrics:
        # take one metric after one another and compute the corresponding piece of code
        if metric=="RMSE":
            mapp = mapping(y_true,y_pred)
            # position of RMSE
            position=np.isin(element=metrics,test_elements="RMSE")
            # compute RMSE
            rmse=RMSE(mu_true,mu_pred,mapp) # RMSE in which a map indicate how points should be associated
            # store RMSE
            res[position]=rmse
#             print('rmse  . = '+str(rmse))
            
        elif metric=="RMSE_ot":
            # position of RMSE
            position=np.isin(element=metrics,test_elements="RMSE")
            # compute RMSE
            rmse=RMSE_ot(mu_true,mu_pred) # RMSE in which the points are automatically assciated so that the distance is minimal
            # store RMSE
            res[position]=rmse
#             print('RMSE_ot = '+str(rmse))
            
        elif metric=="ACC":
            mapp = mapping(y_true,y_pred)
#             print('mapp in ACC'+str(mapp))
            # position of ACC
            position=np.isin(element=metrics,test_elements="ACC")
            # compute ACC
            acc=accuracy(y_true[nb_outliers_true:], y_pred[nb_outliers_true:],mapp)
            # store ACC
            res[position]=acc
            
        elif metric=="ARI":
            # position of ARI
            position=np.isin(element=metrics,test_elements="ARI")
            # compute ARI
            ari=ari(y_true[nb_outliers_true:], y_pred[nb_outliers_true:])
            # store ARI
            res[position]=ari
            
        elif metric=="DISTORSION" or metric=="n_sample":
            # position of DISTORSION
            position=np.isin(element=metrics,test_elements="DISTORSION")
            # a == distorsion value obtained with data that are inliers for both y_pred and y_true against the empirical probability measure Pn (=against data)
            distorsion,inliers_qtty = my_inlier_distorsion(data=x,y_true=y_true,y_pred=y_pred,centers_pred=mu_pred)
            # store DISTORSION
            res[position] = distorsion
            if 'n_sample' in metrics:
                position=np.isin(element=metrics,test_elements='n_sample')
                res[position] = inliers_qtty
        
        elif metric=="OT_CENTERS":
            raise Exception("OT_CENTERS is not yet available in get_metrics")
            mapp = mapping(y_true,y_pred)
            # position of OT_CENTERS
            position=np.isin(element=metrics,test_elements="OT_CENTERS")
            # compute OT_CENTERS
            ot_centers=RMSE(mu_true,mu_pred,mapp)
            # store OT_CENTERS
            res[position]=ot_centers
            
        else:
            raise Exception("the desired metric "+str(metric)+" is not supported in the function get_metrics")
            
    return(res)

def componentwise_format(one_d_array):
    if len(one_d_array)!=2:
        raise Exception('in componentwise_format, the 1d_array must contain 2 values')
        
    res=str(one_d_array[0])+' ('+str(one_d_array[1])+')'
    return(res)

def my_transpose_flatten(two_d_array):
    return(two_d_array.transpose().flatten())
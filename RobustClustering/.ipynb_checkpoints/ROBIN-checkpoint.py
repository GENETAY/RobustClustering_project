from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from scipy.spatial.distance import cdist
import warnings

def dist_to_t(XA,rr):
    '''
    XA = np.array 1D
    rr = np.array 1D
    to fit the shape that arguments will have when using np.apply_along_axis
    '''
    XAA=XA.reshape(1,-1) # convert to 2D array to fit cdist() format
    rrr=rr.reshape(1,-1) # convert to 2D array to fit cdist() format
    res=cdist(XA=XAA,XB=rrr)[0]
    return res

def dist_to_centers(vector,centers):
    '''
    vector  = 1D array
    centers = 1D or 2D array
    returns a 1D array with all distances between the vector and the centers
    '''
    res=np.apply_along_axis(arr=centers,axis=1,func1d=dist_to_t,rr=vector).reshape(1,-1)[0]
    return res

def dist_vec_to_centers(X,centers):
    '''
    X       = 1D or 2D array
    centers = 1D or 2D array
    returns a 1D or 2D array with all distances between each rows of X and the centers
    the rows of the output corresponds to rows of X and column to the centers
    '''
    res=np.apply_along_axis(arr=X,axis=1,func1d=dist_to_centers,centers=centers)
    return res

def ROBIN( X,k,
            method="optimum",epsilon=0.05,determinist=True,
            n_neighbors=20, algorithm='auto', leaf_size=30,
            metric='minkowski', p=2, metric_params=None,
            contamination="auto", novelty=False, n_jobs=None,
            index_first_center = None):
    
    """
    ROBIN is a ROBust INitialization procedure for clustering algorithms that chooses k data as centroids.
    It has been first introduced in Al Hasan et al. (http://twiki.cs.rpi.edu/research/pdf/08-04.pdf).
    
    In their version, the procedure was 100% deterministic which can be problematic when one wants to
    compare the ROBIN approach to others. In this python version, there are 3 possibilities:
    -either choose deterministic=True and index_first_center=None to take the origin of the
    vector space as reference and make procedure unsupervised and deterministic
    -or you can choose deterministic=True and index_first_center=<int> to take the <int>-th data as
    first centroids and make procedure deterministic for the k-1 remaining centroids
    -or you can take deterministic=False and then the first centroids is picked uniformly randomly in the data
    and let the procedure take deterministically the k-1 remaining centroids.
    
    Moreover, there was a algorithmic problem in the original paper: there stopping criterion could be unmet
    resulting in a too small set of centroids (i.e. k was wanted and k-3 were output). this problem was due to
    the fact that they select the next centroids if it has a LOF inside the intervalle ]1-epsilon,1+epsilon[,
    but this may not occur at all if epsilon is too small.
    
    As a correction, we implemented 3 methods:
    -either the original one "approx", that may not found the k centroids. In this case the method goes on with
    the method "optimum" to complete the sequence of k centroids.
    -or the method "optimum" that search in the data for the data point whose LOF is the closest to 1 (with no 
    constrain depending on epsilon).
    -or finally, on can use the method "minimum" which search for the datapoint whose LOF is the smallest among
    all data.
    
    Parameters:
    -----------
        X: 2d numpy array,
            the data point one wants to take k centroids from
        k: int,
            number of data point to return as initialisation centroids 
        method: method to use to select centroids, default "optimum".
            available methods are among ['approx','optimum','minimum']
        epsilon: size of the search intervalle, default 0.05
            the parameter is used only when method='approx'. The method approx
            will search for the datapoint with closest LOF to 1 inside the intervalle
            ]1-epsilon,1+epsilon[ as described in original article.
        determinist: bool, default True
            parameter that indicates whether ROBIN will be deterministic or random.
            when deterministic, a new run will give the same result. No random seed is involved.
            when stochastic, only the first centroid is picked at random, the k-1 others are
            picked according to a deterministic method as described in original article.
        index_first_center: int, default None
            The user can pass in argument the index of the first centroids. This parameter is
            taken into account when deterministic=True.
        
        n_neighbors: one of the parameter of the LOF, default 20.
            see the sklearn.neighbors.LocalOutlierFactor documentation
        algorithm: one of the parameter of the LOF, default 'auto'
            see the sklearn.neighbors.LocalOutlierFactor documentation
        leaf_size: one of the parameter of the LOF, default 30
            see the sklearn.neighbors.LocalOutlierFactor documentation
        metric: one of the parameter of the LOF, default 'minkowski'
            see the sklearn.neighbors.LocalOutlierFactor documentation
        p: one of the parameter of the LOF, default 2
            see the sklearn.neighbors.LocalOutlierFactor documentation
        metric_params: one of the parameter of the LOF, default None,
            see the sklearn.neighbors.LocalOutlierFactor documentation
        contamination: one of the parameter of the LOF, fedault "auto"
            see the sklearn.neighbors.LocalOutlierFactor documentation
        novelty: one of the parameter of the LOF, default False
            see the sklearn.neighbors.LocalOutlierFactor documentation
        n_jobs: one of the parameter of the LOF, default None,
            see the sklearn.neighbors.LocalOutlierFactor documentation
    
    
    """
    
    if not(any(method==np.array(["optimum","minimum","approx"]))):
        print("'method' should be either 'optimum', 'minimum' or 'approx', default 'optimum'. When 'approx' is chosen, give a value for epsilon, default 0.05")
    
    
    # initialization of variables
    n,p=X.shape
    if determinist==True:
        if index_first_center is None:
            # initialize from the origin and the while loop will fill in the centers_index with k centers
            origine=np.tile(A=0,reps=p).reshape(1,p)
            centers_index=[] # we will return the indexes for more convenience
            m=0;m
        else:
            # take the index_first_center as first centroids and the while loop will fill in the centers_index with k-1 centers
            centers=X[index_first_center,:].reshape(1,p)
            centers_index=[index_first_center] # we will return the indexes for more convenience
            m=1;m
    else:
        # take the index_first_center at random and the while loop will fill in the centers_index with k-1 centers
        index_first_center=np.random.choice(np.arange(n),size=1,replace=False)[0]
        centers_index=[index_first_center] # we will return the indexes for more convenience
        centers=X[index_first_center,:].reshape(1,p)
        m=1;m
        
    
    # compute the LOF as first step
    init__=LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size,
                     metric=metric, p=p, metric_params=metric_params,
                     contamination=contamination, novelty=novelty, n_jobs=n_jobs);init__
    LOF_algo_run=init__.fit(X);LOF_algo_run
    LOF_values=-LOF_algo_run.negative_outlier_factor_
    # print("LOF_values="+str(LOF_values))
    
    # search then for k centers
    while m<k:
        #print("m=",m)
        # compute distances to centers except when there is no center yet
        if m==0:
            array_distance=dist_vec_to_centers(X=X,centers=origine).flatten()
        else:
            if m==1:
                array_distance=dist_vec_to_centers(X=X,centers=centers).flatten()
            if m >1:
                array_distance=dist_vec_to_centers(X=X,centers=centers).min(axis=1)
        # find the index order that would sort the distance array
        order=np.flip(m=np.argsort(array_distance),axis=0)
        
        # search for the next center
        if method=='optimum':
            # Looks for the LOF that is the nearest to 1: one transforms the LOF_values with x->(x-1)^2 and then searchs the min
            position_ordered_LOF_next_nearest_to_1=np.argsort(((LOF_values[order]-1)**2))[m];position_ordered_LOF_next_nearest_to_1
            position_LOF_next_nearest_to_1=order[position_ordered_LOF_next_nearest_to_1];position_LOF_next_nearest_to_1
            centers_index.append(position_LOF_next_nearest_to_1);centers_index
            point_LOF_next_nearest_to_1=X[position_LOF_next_nearest_to_1,:];point_LOF_next_nearest_to_1
            if m==0:
                centers=point_LOF_next_nearest_to_1.reshape(1,p);#print("path 2.1 : ",centers)
            else:
                centers=np.concatenate([centers,point_LOF_next_nearest_to_1.reshape(1,p)]);#print("path 2.2 : ",centers)
        elif method=='minimum':
            # Looks for the LOF that is minimum
            position_ordered_LOF_next_min=np.argsort(LOF_values[order])[m];position_ordered_LOF_next_min
            position_LOF_next_min=order[position_ordered_LOF_next_min];position_LOF_next_min
            centers_index.append(position_LOF_next_min)
            point_LOF_next_min=X[position_LOF_next_min,:];point_LOF_next_min
            if m==0:
                centers=point_LOF_next_min.reshape(1,p)
            else:
                centers=np.concatenate([centers,point_LOF_next_min.reshape(1,p)])
        elif method=="approx":
            # Looks for the first LOF value that falls in ]1-eps,1+eps[
            i=0
            stopping_crit=True
            while stopping_crit:
                if i<n:
                    # it is indeed possible that one has pointed at all LOF values while centers are still missing.
                    # this situation means that the intervalle ]1-eps,1+eps[ contains too few points to be centers.
#                     print(i,end='')
                    if abs(LOF_values[order[i]]-1)<epsilon and not(any(order[i]==centers_index)):
                        centers_index.append(order[i])
                        if m==0:
                                centers=X[order[i],:].reshape(1,p)
                        else:
                                centers=np.concatenate([centers,X[order[i],:].reshape(1,p)])
                        stopping_crit=False
                    else:
                        i=i+1
                else:
                    # all values of LOF have been pointed by the loop and it still misses centers then one uses the 'optimum' method to complet centers 
                    stopping_crit=False
                    method="optimum"
                    warnings.warn('ROBIN could not find any data with LOF inside intervalle ]1-epsilon,1+epsilon[ during procedure. ROBIN goes on with method "optimum. Increase epsilon or decrease k if you do not want this message to show up."')
        else:
            print("warnings:: unvalid value given for 'method'")
    
        # we now have m+1 centers, we need to increment m
        m=len(centers_index);m

    res=np.array(centers_index)
    return res
from random import sample
import numpy as np

def trimmed_Kmeans(data,k,trim=0.1, runs=100, points= None,printcrit=False,maxit=None):
    '''
    Trimmed kmans is a robust clustering algorithm that outputs k centroids. It optimizes
    these centroids as kmeans does but the proportion "trim" of the data that are the
    furthest from the centroids are ignored during the maximization step of the EM algo
    (i.e. the new mean vectors are computed without these trimmed data point.)
    
    trimmed kmeans is available in R language: https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html
    and it was not in 2019 went we needed it so we made it available in python.
    Since then, some codes where delivered:
    - https://github.com/OlgaD400/Robust-Trimmed-K-Means
    
    trimmed-kmeans must not to be confounded with trim-means that does a mean of real value
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html)
    
    Parameter:
    ----------
        data: np.array
            the dataset
        k : int,
            number of clusters (or centroids)
        trim: float between 0 and 1, default 0.1
            trimmed proportion of data
        runs: int, default 100
            maximal number of iterations. If this criterion is not met, this means that the
            algorithm stopped because the partition had converged to an equilibrium (no more
            changes in the partition from an iteration to another)
        points : numpy array, default None,
            initial datapoints.
    Return:
    -------
        a dictionary with keys ['classification', 'means', 'criterion', 'disttom', 'ropt',
        'k', 'trim', 'runs'].
        
        classification:
            the numpy array with the assignment of each data to a cluster.
            -1 means "trimmed data"
        means:
            the means vectors (1st center is output['means'][0])
        criterion:
            the value of the kmeans distorsion computed on non-trimmed data only
        disttom:
            means 'distance to means', this is the array with all distances of the
            datapoints to their closest centroids
        ropt:
            biggest distance between an inlier and its centroids (inlier = not outlier)
        k:
            input parameter, number of clusters
        trim:
            input parameter, proportion of data to trimm
        runs:
            input parameter, maximal number of iterations
    '''
    if maxit is None:
        maxit = 2*len(data)
    
    countmode = runs+1
    data = np.array(data)
    n,p  = data.shape
    nin  = round((1-trim)*n)
    crit = np.Inf

    oldclass = np.zeros((n,))
    iclass   = np.zeros((n,))
    optclass = np.zeros((n,))
    disttom  = np.zeros((n,))
    
    for i in range(runs):
        #if i/countmode == round(i/countmode):
            #print("iteration",i)
        if points is None:
            means = data[sample(np.arange(n).tolist(),k),:]
        else:
            means = points.copy()
        wend = False
        itcounter = 0

        while not wend:
            itcounter += 1
            for j in range(n):
                dj = np.zeros((k,))
                for l in range(k):
                    #print(data[j,:],means[j,:])
                    dj_   = (data[j,:]-means[l,:])**2
                    dj[l] = dj_.sum()
                iclass[j] = dj.argmin()
                disttom[j]= dj.min()

            order_idx = np.argsort(disttom)[(nin+1):] # trimmed ones
            iclass[order_idx] = -1
            
            if itcounter >= maxit or np.all(oldclass in iclass) :
                wend = True
            else:
                for l in range(k):
                    if sum(iclass==l)==0 : # to my mind, if the condition is met then class l is empty and mean vector makes no sense.
                        means[l,:] = data[iclass==0,:]
                    else:
                        if sum(iclass==l)>1 : # if >1 then numpy array is a 2d array
                            if means.shape[1] == 1:
                                means[l,:] = data[iclass==l,:].means()
                            else:
                                means[l,:] = data[iclass==l,:].means(axis=1)
                        else: # otherwise ==1 then numpy array is a 1d array
                            means[l,:] = data[iclass==l,:]
                oldclass = iclass
        
        newcrit = disttom[iclass >= 0].sum()
        if printcrit:
            print("Iteration ",i," criterion value on non-trimmed data is ",newcrit/nin,"\n")
        if newcrit <= crit :
            optclass = iclass.copy()
            crit = newcrit.copy()
            optmeans = means.copy()
            
    out = {'classification':optclass,'means':optmeans,'criterion':crit/nin,'disttom':disttom,
           'ropt':np.sort(disttom)[nin],'k':k,'trim':trim,"runs":runs}
    return out
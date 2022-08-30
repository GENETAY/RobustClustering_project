# Date : 2022-08-25

# !/usr/bin/python
# -*- coding: utf-8 -*-
# """
# K-bMOM algorithm
# """

__author__ = "Saumard Adrien, Camille Saumard, Edouard GENETAY"
__copyright__ = "Copyright 2022"
__version__ = "8.5"
__maintainer__ = "Edouard GENETAY"
__email__  = "genetay.edouard@gmail.com"
__status__ = "version beta"


import numpy as np
from numpy import nan
import warnings
import random as random_pck # not to be confused with numpy.random
from math import log,inf,floor,ceil

from scipy.spatial.distance import cdist
from .kmedianpp import kmedianpp_init
from .utils import flatten_list

class KbMOM:
    """
    A class of robust cluster estimator named KbMOM in honor of its first name in the literature: K-bMOM.
    
    Remark
    ------
    The KbMOM method can theoretically perform parallel computation on the blocks but it is not implemented yet.

    Attributes
    ----------
    X : 2d numpy array, no default
            data we want to cluster. Given at instantiation to know their number and their dimensionality.
            they are not stored so they should be passed again in method fit().
        K : int, no default
            number of clusters
        number_of_blocks : int, no default
            number of blocks to be created in the initialization and iterative part
        block_size : int, no default
            NUMBER of data in each block and cluster
        max_iter : int, default 20
            number of iterations of the algorithm
        q_quantile : float, default 0.5.
            wanted quantile to select one block among all. It risk will be the empirical selected risk. by default q_quantile=0.5 and corresponds to the median.
        estimated_number_of_outliers : int, default None.
            assumed or estimated number of outliers. If this number is not known, you take a greater number than data seems to show in order to be safe in estimation.
        confidence : float, default 0.95
            maximal probability that the median block is corruption free. If you are dealing with a lot of corruption, KbMOM will not prevent the quantile block to be corrupted,
            it will try to make it clean at least with probability "confidence". It can be useful to clean the data to improve estimation. Think of using the data_depth
            in the output of the fit method.
        Aitkens_criterion_allowed: bool, optional, default False.
            Enable or not to use the Aitkens criterion to stop the iteration part of KbMOM. If False, the KbMOM will do max_iter iterations.
        Aitkens_threshold : float, default 0.00001
            Threshold used with the aitkens criterion. This threshold is so small that it will not be reached before the maximal number of iterations are done.
            Usually clustering methods use a convergence criterion because some quantity decrease long iterations. In KbMOM though, the centroids are selected through a
            stochastic process and no quantity decreases at each iteration. It may decrease in average, it remains always some fluctuation though.
            We implemented such a criterion for those that would like to test it. But it was meant to work with a maximal number of iteration.
        init_by_kmeanspp : bool, default True.
            This parameter controls whether our own initializes method will use kmeans++ or kmedians++. Although kmeans++ and kmedians++ are not robust initialization,
            our initialization procedure based on them is robust. see our article Genetay, Saumard, Saumard to see to what extend it is robust. We advize you to use
            kmeans++ because it has shown better results in our experiments.
        nb_min_repr_by_cluster : int, default 2.
            how many representant of each cluster there must at least be. This constrain has no effect at the initialization step because there is no assignment yet.
            This is applied in the iteration steps. One can naturally choose 0 or 1.
            -If 0 is chosen then some blocks can have empty clusters. These blocks with empty clusters can be the one achieve the quantile value of the risk and then
            be chosen to recompute the centroids. In this case, the concerned centroids will disapear and the process go on with a smaller K.
            -If 1 is chosen, then all clusters will have at least 1 element in each blocks. It can happen that the KbMOM algorithm get stuck in the configuration where
            one centroids is located on an outlier. If you fear such a situation has happened, then you can see where the centroids of all iteration were located in the
            output of the method fit().
        data_depth_strategy : str, default "block_members_distances".
            This variable controls how the depth is computed. It can take three values from the list ["denumber", "centroids_distances", "block_members_distances"].
            
            There are 3 intuitive data depth to be built from the block selection:
            1-Either denumber the number of time a datapoint is in the selected block by the
            q_quantile seclection
            2-Either keep track of a measure of how close the datapoints are from the centroids
            inside the selected block
            3-Or keep track of a measure of how close the data points are from the elements of
            the selected block.

            This feature was not described in the article of Genetay, Saumard, Samard because
            the article was already long and because more experiments should have been done to
            compare the performances of our depth to the other depth of the literature. It was
            though implemented. feel free to use it and to compare it to other depths. Feel even
            free to publish an article about it, it won't be done by Genetay and al.
        random_state: int, default None.
            set the random seed. Note that all randomness relies on package "random" because numpy.random is unstable through threads. To this end, auxiliary code
            (for initialization kmeanspp for example) found on the internet has been manually modified also.

    Methods
    -------
    sampling_init:
        create block at initialization step
    sampling_all_blocks:
        create block at iteration step
    within_var:
        compute inertia of clusters
    find_the_q_quantile_risk_and_block:
        compute the risks of all blocks and selects the block that achieve the wanted quantile
    compute_quantile_block_centers:
        compute centers inside the block achieving the quantile of the risks among all blocks
    shape_partition_into_dictionary:
        transforms partition array into a cluster composition dictionary
    compute_data_depth:
        compute a data depth among three custom data depth. These data depth use the quantile block to extrapolate centrality to all data.
    update:
        update several internal variable after having selected the quantile block
    initialisation_without_init_centers:
        perform the special robust KbMOM-like initialization.
    initialisation_with_init_centers:
        uses centroids passed in agrument to update internal necessary quantities
    predict:
        return the assignment array (a.k.a. partition array)
    fit:
        perform initialization and iterations in KbMOM
    get_centers:
        return the current centroids in memory
    take_q_quantile_with_negative_pollution:
        compute a custom quantile function that ignore negative values
    block_size_max:
        return the maximal block size that ensures that the quantile block is corruption-free with probability at least confidence
    minimal_number_of_blocks:
        return the minimal number of blocks that makes possible that the quantile block is corruption-free with probability at least confidence
    stopping_crit:
        compute the Aitkens stopping criterion
    stopping_crit_GMM:
        compute the same stopping criterion as in GMM algorithm
    """
    def __init__(
        self,
        X,
        K,
        number_of_blocks,
        block_size,
        max_iter = 20,
        q_quantile = 0.5,
        estimated_number_of_outliers = None,
        confidence = 0.95,
        Aitkens_criterion_allowed = False,
        Aitkens_threshold = 0.00001,
        initial_centers = None,
        init_by_kmeanspp = True,
        nb_min_repr_by_cluster = 2,
        data_depth_strategy="block_members_distances",
        random_state = None
        ):
        """
        Constructs all the necessary attributes for the KbMOM object.

        Parameters
        ----------        
            all attributes of the class
        """
        
        # given element
        self.K = K
        self.max_iter = max_iter
        self.n, self.p = X.shape
        self.q_quantile = q_quantile
        self.block_size = block_size
        self.number_of_blocks = number_of_blocks
        self.alpha = 1 - confidence
        self.Aitkens_threshold = Aitkens_threshold
        self.init_by_kmeanspp = init_by_kmeanspp
        self.nb_min_repr_by_cluster = nb_min_repr_by_cluster
        self.data_depth_strategy = data_depth_strategy
        self.random_state = random_state
        self.initial_centers = initial_centers
        self.Aitkens_criterion_allowed = Aitkens_criterion_allowed
        
        # other instantiation
        self.data_depth = np.zeros((self.n,))
        self.centers = None
        self.iter = 0
        if self.random_state is None:
            random_pck.seed(self.random_state)
        self.has_model_already_been_fitted = False
        
        # Test some given values
        if estimated_number_of_outliers is not None:
            self.estimated_number_of_outliers = estimated_number_of_outliers
            minimal_number_of_blocks_ = KbMOM.minimal_number_of_blocks(
                n=self.n,
                nb_outliers=self.estimated_number_of_outliers,
                block_size=self.block_size,
                alpha=self.alpha
            )
            if self.number_of_blocks < minimal_number_of_blocks_ :
                self.number_of_blocks     = int(round(minimal_number_of_blocks_) + 10)
                warnings.warn('the number of blocks has been computed according to the breakdown point theory')

            block_size_sup_bound = KbMOM.block_size_max(
                n=self.n,
                nb_outliers=self.estimated_number_of_outliers
            )
            if self.block_size > block_size_sup_bound :
                self.block_size = int(round(max((t_sup-5),1)))
                warnings.warn('the size of blocks has been computed according to the breakdown point theory')
            
        
        # Deal with exceptions:
        if self.n < self.nb_min_repr_by_cluster*self.K:
            line_1 = "either K or nb_min_repr_by_cluster is too big. they must satisfy K*nb_min_repr_by_cluster<=n, "
            line_2 = "with n the size of the data sample in order block to be at most as big as the whole sample"
            raise Exception(line_1+line_2)
            
        if self.block_size < self.nb_min_repr_by_cluster*self.K:
            self.block_size = self.nb_min_repr_by_cluster*self.K
            warnings.warn(
                'block_size is too small considering that you want K clusters with at least nb_min_repr_by_cluster elements per cluster.Then block_size has been set to K*nb_min_repr_by_cluster'
            )
            
    
    def sampling_init(self,X):
        """Create number_of_blocks blocks initialized with a kmeans++ or a kmedian++ (depends on the boolean value of self.init_by_kmeanspp)."""
        initial_blocks = [0]*self.number_of_blocks # instantiation
        
        # for instanciation of kmeans++
        x_squared = X**2
        x_squared_norms = x_squared.sum(axis=1)
        
        # Blocks creation
        size_of_blocks = self.block_size
        for i in range(self.number_of_blocks):
            idx = random_pck.choices(np.arange(self.n),k = int(size_of_blocks))
            
            centroids_from_non_robust_initialisation = kmedianpp_init( # either kmedian++ or kmeans++ depending on the value of squared
                X[idx,:], 
                self.K, 
                x_squared_norms[idx],
                random_state = 2*self.random_state if self.random_state is not None else self.random_state, 
                n_local_trials = None,
                squared = self.init_by_kmeanspp
            )
            distance_data_to_centroids_in_the_block = cdist(X[idx,:],centroids_from_non_robust_initialisation)
            assignment_data_in_block_to_nearest_point_centroids = distance_data_to_centroids_in_the_block.argmin(axis=1)
            
            initial_blocks[i] = [[j for j,x in zip(idx,assignment_data_in_block_to_nearest_point_centroids) if x==k] for k in range(self.K)]
            
        return initial_blocks
            
        
    def sampling_all_blocks(self,X):
        """
        Function that creates number_of_blocks blocks when data are already assigned to their nearest centroids.
        With this assignment knowledge, the function also makes sure that each cluster in each block contains at
        least nb_min_repr_by_cluster elements.
        
        Take nb_min_repr_by_cluster as big as you can to improve the quality of the estimation of the centroids
        in the data.
        
        if nb_min_repr_by_cluster == 0, then some cluster can be empty and their will be some blocks where it will
        be impossible to compute K centroids. The code handle this situation and performs the selection of the blocks
        as usual. A block with strictly less than K centroids can be selected if it achieve the q_quantile value of
        the risk and the process goes on with a assignment vector that contains strictly less that K classes.
        
        To prevent the situation from happening, take nb_min_repr_by_cluster >= 1
        
        Parameters
        ----------
        X : 2d numpy array, no default.
            datapoints to be clustered
       
        Returns
        -------
        list_of_blocks : list
            list of lists containing indexes of the datapoints in X forming the block
        """
        list_of_blocks = [0]*self.number_of_blocks
        for i in range(self.number_of_blocks):
            # to ensure there are K categories in each block run that part, one can go through the first piece of code, else blocks are completely random
            list_of_points = []
            if self.nb_min_repr_by_cluster is not None:
                for k in range(self.K):
                    list_of_points = list_of_points + random_pck.choices(np.array(self.partition_dict[k]), k=self.nb_min_repr_by_cluster)
                idx_indep = random_pck.choices(np.arange(self.n), k=self.block_size-self.nb_min_repr_by_cluster*self.K)
            else:
                idx_indep = random_pck.choices(np.arange(self.n), k=self.block_size)
            idx = list_of_points + idx_indep

            distance_data_to_centroids_in_the_block = cdist(X[idx,:],self.centers)
            assignment_data_in_block_to_nearest_point_centroids = distance_data_to_centroids_in_the_block.argmin(axis=1)         
            list_of_blocks[i] = [[j for j,x in zip(idx,assignment_data_in_block_to_nearest_point_centroids) if x==k] for k in range(self.K)]
            
        return list_of_blocks
    
    def within_var(self,one_block,X,nb_min_repr_by_cluster):
        """return a list of within inertia (a.k.a. within variance) per cluster from a block partitioned into clusters"""
        within_var_per_cluster = [0]*self.K

        if sum(np.array(list(map(len,one_block))) >= nb_min_repr_by_cluster) == self.K:
            for key , cluster_ids in enumerate(one_block):
                cluster = X[cluster_ids,:]
                within_var_in_cluster = len(cluster)*np.sum(np.var(cluster,axis=0))/self.block_size
                within_var_per_cluster[key] = 0.0 if np.isnan(within_var_in_cluster) else within_var_in_cluster
                
                # Why is there "np.isnan()" here?
                # because in the case where nb_min_repr_by_cluster = 0, there is no constrain on the size of a cluster.
                # it can then happen that a cluster is empty leading to an undefined value for within_var_in_cluster.
                # Since, no constrain was required, such a block is valid though, replace the "np.nan" by a "0.0"
                
        else:
            # store a value that will be useful to discard the block
            within_var_per_cluster = [-1]
            
        return within_var_per_cluster
    
            
    def find_the_q_quantile_risk_and_block(self,all_blocks,X,nb_min_repr_by_cluster):
        """
        This function computes the risk in all_blocks and searchs among all_blocks
        which one achieves the value corresponding to the wanted quantile q.
        
        Parameters
        ----------
        all_blocks : list, no default
            List of lists (blocks) containing K lists (clusters).
        X : 2d numpy array, no default
            Datapoints to be clustered
        nb_min_repr_by_cluster : int, default 2.
            how many representant of each cluster there must at least be.
            This constrain has no effect at the initialization step because
            there is no assignment yet.
        
        Returns
        -------
        risk_in_selected_block : 
            risk value corresponding to the quantile q among the risks of all blocks
        within_variances_inside_selected_block : 
            inertia of each cluster inside the block that achieve the quantile q of the risk
        selected_block :
            the block inside all_blocks that achieve the quantile q of the risk
        dict_info :
            dictionary indicating how many blocks were discarding because of clusters with too
            small size compared to the minimum requested (nb_min_repr_by_cluster).
            dictionary has following format:
            {'info': "'x' blocks have been discraded"}
        """

        list_all_within_variances = [0]*self.number_of_blocks
        list_all_risks_per_block = [0]*self.number_of_blocks

        # compute the kmeans losses
        for key, one_block_ in enumerate(all_blocks):
            list_all_within_variances[key] = self.within_var(one_block_,X,nb_min_repr_by_cluster)
            # the one that do not satisfy the constrained imposed by nb_min_repr_by_cluster have a within_var of [-1]
            list_all_risks_per_block[key] = sum(list_all_within_variances[key])


        # Now select the block that achieve de quantile q of the risks while ignoring the -1 values
        array_all_risks_per_block = np.array(list_all_risks_per_block)
        q_quantile_position_skipping_minus_ones, risk_in_selected_block, number_of_minus_one = KbMOM.take_q_quantile_with_negative_pollution(array_all_risks_per_block,self.q_quantile)
        dict_info = {'info': f"{number_of_minus_one} blocks have been discraded"}
        
        # extract interesting objects
        selected_block = all_blocks[q_quantile_position_skipping_minus_ones]
        within_variances_inside_selected_block = list_all_within_variances[q_quantile_position_skipping_minus_ones]    

        # save the selected risk to be albe to monitor it in post-analysis
        self.list_all_q_quantile_risks_through_iterations.append(risk_in_selected_block)

        return risk_in_selected_block,within_variances_inside_selected_block,selected_block,dict_info

    
    def compute_quantile_block_centers(self,q_quantile_block,X):
        """return the mean vector of each cluster in the q_quantile block as 2d numpy array"""
        block_centroids = []
        for k in range(self.K):
            center_k = X[q_quantile_block[k],:].mean(axis=0)
            block_centroids.append(center_k)
        return np.array(block_centroids)
    
    def shape_partition_into_dictionary(self,partition):
        """
        Function which shapes the partition array or list as dictionary
        where keys are cluster id and value are lists of data ids.
        
        Example
        -------
            context:
                data = [[0,0],[0,1],[5,0],[5,1],[5,2]]
                K (number of clusters) = 2
                centroids = [[0,0.5],[5,1]]
                partition = [0,0,1,1,1] or np.array([0,0,1,1,1])
                
            wanted transformation:
                partition_dict will be udpated as
                    {0:[0,1], 
                     1:[2,3,4]}
                because data points 0 and 1 and in the cluster 0 and data points 2,3 and 4 and in cluster 1.
        
        Parameters
        ----------
        partition : list, no default
            list of datapoints affectations. partition[i] indicates the cluster of i-th datapoint.
        """
        partition_dict = {k:[] for k in range(self.K)}
        for i, x in enumerate(partition):
            partition_dict[x].append(i)
        self.partition_dict = partition_dict
    
    def compute_data_depth(self,X,quantile_block,quantile_block_within_variances,distances_data_to_its_centroid,partition_array):
        """
        Function which updates the data depth of the datapoints.
        
        There are 3 intuitive data depth to be built from the block selection:
        1-Either denumber the number of time a datapoint is in the selected block by the
        q_quantile seclection
        2-Either keep track of a measure of how close the datapoints are from the centroids
        inside the selected block
        3-Or keep track of a measure of how close the data points are from the elements of
        the selected block.
        
        The depth is choosen through an argument at the instantiation called "data_depth_strategy".
        Its value should be taken in the following list ["denumber", "centroids_distances",
        "block_members_distances"].
        
        This feature was not described in the article of Genetay, Saumard, Samard because
        the article was already long and because more experiments should have been done to
        compare the performances of our depth to the other depth of the literature. It was
        though implemented. feel free to use it and to compare it to other depths. Feel even
        free to publish an article about it, it won't be done by Genetay and al.
        
        Parameters
        ----------
            X : 2d numpy array, no default
                datapoints to be clustered
            quantile_block :
                block that achieves the wanted quantile of the risks values
            quantile_block_within_variances: list of K non negative floats, no default.
                list of inertia in each cluster of the quantile block
            distances_data_to_its_centroid: numpy array
                gives the distance between the datapoints X of their nearest centroids
        """
        if self.data_depth_strategy == 'denumber':
            # idea:
            # consider each data is more central when it is often in the select block (quantile q block)
            # works if the block size is big compared to the number of data
            for clus, idk in enumerate(quantile_block):
                self.data_depth[idk] += 1
                
        elif self.data_depth_strategy == 'centroids_distances':
            # idea:
            # consider each data is more central when it is often close to the selected block centroids (quantile q block centroids)
            # work always. A slight flaw is that it tends to add a spherical symmetry in the depth
            for clus, idk in enumerate(quantile_block):
                if quantile_block_within_variances[clus] == 0:
                    # not to divide by zero
                    self.data_depth[idk] += 1 # This 1 comes not from a limit computation in next formula, it is just the biggest value possible of the next formula.
                else:
                    self.data_depth[idk] += np.exp(-distances_data_to_its_centroid[idk]/quantile_block_within_variances[clus]) # converges to finite value when quantile_block_within_variances[clus] tends to 0
        
        elif self.data_depth_strategy == 'block_members_distances':
            # idea:
            # consider each data is more central when it is often close to the points inside the selected block (quantile q block centroids
            # work always.
            elements_in_Qblock = X[flatten_list(quantile_block)]
            matrice_distances_data_centroids = cdist(XA=X,XB=elements_in_Qblock) # distance between datapoint and those in quantile_block
            distances_data_to_its_centroid = np.amin(matrice_distances_data_centroids,axis=1) # distance between datapoint and the nearest data in quantile_block
            for cluster_id in range(self.K):
                mask_cluster = partition_array == cluster_id
                if quantile_block_within_variances[cluster_id] == 0:
                    # not to divide by zero
                    self.data_depth[mask_cluster] += 1
                else:
                    self.data_depth[mask_cluster] += np.exp(-distances_data_to_its_centroid[mask_cluster]/quantile_block_within_variances[cluster_id])
        else:
            raise Exception('data_depth_strategy should be one of the string in ["denumber", "centroids_distances", "block_members_distances"]')
            
    
    def update(self,X,quantile_block,quantile_block_within_variances,nb_min_repr_by_cluster):
        """
        Function which updates current centroids, assignment vector (partition of data), size of clusters.

        In addition it updates the data_depth that are custom feature that were tested but not reported
        in the article of Genetay, Saumard and Saumard due to no improvement on results.

        X :
           data to be clustered 
        quantile_block :
           the block achieving the quantile q of the risks among blocks.
           i.e. the median if q=0.5
        quantile_block_within_variances :
           list of all within inertia of each cluster in the quantile_block
        nb_min_repr_by_cluster : int, default 2.
            how many representant of each cluster there must at least be.
            This constrain has no effect at the initialization step because
            there is no assignment yet.
        
        Parameters
        ----------
            X : 2d numpy array, no default
                datapoints to be clustered
            quantile_block :
                block that achieves the wanted quantile of the risks values
            quantile_block_within_variances: list of K non negative floats, no default.
                list of inertia in each cluster of the quantile block
        """
        # updates centers
        self.centers = self.compute_quantile_block_centers(quantile_block,X)
        self.list_of_all_quantile_block_centers_through_iterations.append(self.centers)

        # retrieve partition of data
        matrice_distances_data_centroids = cdist(XA=X,XB=self.centers) # take distances from centroids
        assignment_data_to_nearest_centroid = matrice_distances_data_centroids.argmin(axis=1)

        # compute empirical risk
        distances_data_to_its_centroid = np.amin(matrice_distances_data_centroids,axis=1) # minimum along data
        empirical_risk = (distances_data_to_its_centroid**2).mean()
        self.list_all_empirical_risks_through_iterations.append(empirical_risk)

        # update size of clusters:
        self.cluster_size = np.bincount(assignment_data_to_nearest_centroid)

        # compute the weights of each point:
        self.compute_data_depth(X
                                ,quantile_block
                                ,quantile_block_within_variances
                                ,distances_data_to_its_centroid
                                ,assignment_data_to_nearest_centroid
                               )

        # change format of affectation vector (from array to list)
        self.shape_partition_into_dictionary(assignment_data_to_nearest_centroid)
    
    
    def initialisation_without_init_centers(self,X):
        """This function computes the first centroids and initializes the partition of the data."""
        # initialisation per block: sampling M blocks and init via kmeans++
        init_blocks = self.sampling_init(X)
        
        # compute empirical risk among blocks and select the Q-quantile-block
        # in the initialization, the value nb_min_repr_by_cluster must be zero
        # because nb_min_repr_by_cluster bring a constrain for the iterations only
        _,q_quantile_block_within_variances,q_quantile_block,_ = self.find_the_q_quantile_risk_and_block(init_blocks,X,nb_min_repr_by_cluster=0)
        
        # update all the global variables
        self.update(X,q_quantile_block,q_quantile_block_within_variances,nb_min_repr_by_cluster=0)
        
        # save results
        self.q_quantile_block_at_initialization = q_quantile_block
        
        
    
    def initialisation_with_init_centers(self,X):
        """Standard initialization cannot handle known centroids. This function goes manually through all necessary computations to go on as usual"""
        # one cannot use the routine initialize centroids, then find quantile risk and update
        # because it can only be done when centers are not known. In this case, knowing
        # the centers goes more straitghfowardly to computations of partition and empirical risk
        # without updating the values of some quantities such as centers, distance_matrix, etcs.
        # So we do it manually to be able to go on with the usual routine.
        
        # take initial centers given as parameter
        self.centers = self.initial_centers
        
        # retrieve partition
        matrice_distances_data_centroids = cdist(XA=X,XB=self.centers)
        assignment_data_to_nearest_centroid = matrice_distances_data_centroids.argmin(axis=1)
        
        # compute empirical risk
        distance_data_to_nearest_centroid = matrice_distances_data_centroids[[np.arange(self.n).tolist(),assignment_data_to_nearest_centroid.tolist()]]
        empirical_risk = (distance_data_to_nearest_centroid**2).mean()
        
        # compute cluster sizes
        self.cluster_size = np.bincount(assignment_data_to_nearest_centroid)
        
        # update the value of the internal variable partition_dict
        self.shape_partition_into_dictionary(assignment_data_to_nearest_centroid)
        
        # save results
        # the q_quantile_risk of the block does not exist when init centers are given, so we give it infinite value, so that risk decreases
        self.list_all_q_quantile_risks_through_iterations.append(inf)
        self.list_all_empirical_risks_through_iterations.append(empirical_risk)
        self.list_of_all_quantile_block_centers_through_iterations.append(self.centers)
    
    def predict(self,X):
        '''
        Function which computes the partition based on the last centroids in memory.
        The last centroids in memory are either the initial centers if iter_max is 0
        otherwise the are the centroids of the last quantile block.
        
        Use fit before calling this method, otherwise centers is None and the prediction cannot be made.
        '''
        if self.centers is None:
            raise Exception('call fit method before calling the predict method because the internal variable "center" is currently None.')
        
        matrice_distances_data_centroids = cdist(XA=X,XB=self.centers)
        return matrice_distances_data_centroids.argmin(axis=1)
    
    def fit(self,X):
        """Main loop of the K-bMOM algorithm:"""
        # if the model has already been fitted, return computed output
        if self.has_model_already_been_fitted:
            return self.fit_output
        
        # else continue
        # instantiate useful variables
        self.q_quantile_block_at_initialization = None
        self.list_all_q_quantile_risks_through_iterations = []
        self.list_all_empirical_risks_through_iterations = []
        self.list_of_all_quantile_block_centers_through_iterations = []
        Aitkens = [None,None]
         
        # initialisation part
        if self.initial_centers is not None:
            self.initialisation_with_init_centers(X)
        else:
            self.initialisation_without_init_centers(X)
        
        # iteration part
        if (self.max_iter == 0):
            condition = False
        else:
            condition = True
            
        while condition:
            # sampling
            all_blocks = self.sampling_all_blocks(X)
            
            # Compute empirical risk for all blocks and select the block achieve the q_quantile of the risk
            _,q_quantile_block_within_variances,q_quantile_block,_ = self.find_the_q_quantile_risk_and_block(all_blocks,X,self.nb_min_repr_by_cluster)      
            
            # updates
            self.update(X,q_quantile_block,q_quantile_block_within_variances,self.nb_min_repr_by_cluster)
            
            self.iter += 1
            if self.iter>1 and self.Aitkens_criterion_allowed:
                Aitkens_ = KbMOM.stopping_crit(self.list_all_q_quantile_risks_through_iterations)
                Aitkens.append(Aitkens_)
                if Aitkens_ < self.Aitkens_threshold:
                    condition = False
            if self.iter >= self.max_iter:
                condition = False
                
        
        self.fit_output = {'centroids':self.centers
                ,'labels':self.predict(X)
                ,'clusters_composition':self.partition_dict
                ,'q_quantile_block_at_initialization':self.q_quantile_block_at_initialization
                ,'convergence': Aitkens
                ,'data_depth':self.data_depth
                ,'all_data_risks': self.list_all_empirical_risks_through_iterations
                ,'quantile_block_risks': self.list_all_q_quantile_risks_through_iterations
                ,'list_of_all_quantile_block_centers_through_iterations': self.list_of_all_quantile_block_centers_through_iterations
                ,'number_of_blocks':self.number_of_blocks
                ,'block_size':self.block_size
                ,'n_iter':self.iter
               }
        
        self.has_model_already_been_fitted = True
        return self.fit_output
    
    def get_centers(self):
        return self.centers
    

    def take_q_quantile_with_negative_pollution(numpy_array,q_quantile):
        """
        This function compute any quantile quantile of positive values in presence of negative values.
        The quantile are not uniquely defined that is why this function always outputs the smallest
        candidate for a quantile (i.e. the median of [1,2,3,4,5,6] is 3 and the 3rd quartile is 4 while
        people often choose 3.5 = (3+4)/2 as median and 4.5=(4+5)/2 or 5 as quartile.)
        
        Parameters
        ----------
            numpy_array : 1d numpy array, no default
                array of number to extract a specified quantile q from
            q_quantile : float, no default.
                number between 0 and 1 sctrictly. Indicates the wanted quantile.
                0.5 corresponds to the median.                
        
        Return
        ------
            q_quantile_position_skipping_minus_ones :
                the position of the q_quantile value in the numpy_array
            q_quantile_value_skipping_minus_ones :
                the value of the q_quantile
            number_of_minus_one :
                the number of negative values that were ignored in the numpy_array
        """
        sample_size = len(numpy_array)
        number_of_minus_one = sum(numpy_array < 0)    
        
        array_argsort = np.argsort(numpy_array,kind="mergesort") # sort all values, including -1's
        quantile_among_kept_values = int(np.floor(q_quantile*(sample_size-number_of_minus_one-1)))
        quantile_among_all_values = number_of_minus_one + quantile_among_kept_values
        
        q_quantile_position_skipping_minus_ones = array_argsort[quantile_among_all_values]
        q_quantile_value_skipping_minus_ones = numpy_array[q_quantile_position_skipping_minus_ones]
        
        return q_quantile_position_skipping_minus_ones, q_quantile_value_skipping_minus_ones, number_of_minus_one
    
    def block_size_max(n,nb_outliers):
        '''
        Function which gives the maximum allowed size of blocks before the breakpoint of the method.

        Note: The method estimates the mean of a distribution, this method is robust to outliers to
        some extent. If the number of outlier is too big, the breakdown point is reached and the method
        do not estimated the mean anymore.

        ```prms```
        n: nb of data
        n_outlier: nb of outliers
        '''
        outlier_proportion = nb_outliers/n

        if nb_outliers == 0:
            bloc_size_max = inf
        else:
            bloc_size_max = floor(log(2.)/log(1/(1-outlier_proportion)))
        if bloc_size_max == 0:
            warnings.warn("in function block_size_max gave back a maximal block size of zero. This may impact the following computations. This error comes out because there is more corrupted data than regular ones.")
        return bloc_size_max


    def minimal_number_of_blocks(n,nb_outliers,block_size=None,alpha=0.05):
        '''
        Function which fits the minimum nb of blocks for a given size t before a the breakpoint
        ```prms```
        n: nb of data
        nb_outliers: nb of outliers
        block_size = bloc_size
        alpha : confidence threshold
        '''
        outlier_proportion = nb_outliers/n

        if nb_outliers/n >= 0.5:
            warnings.warn('Either the number of outliers is too big or there is too much noise in the data, the maximal number of block to match the conditions is then undefined. The function minimal_number_of_blocks outputs "None"')
            
        if nb_outliers == 0 :
            bloc_nb_min = 1
        else:
            if block_size is not None:
                if (1-outlier_proportion)**block_size - 1/2 > 0:
                    bloc_nb_min = ceil(log(1/alpha) / (2* ((1-outlier_proportion)**block_size - 1/2)**2))
                else:
                    warnings.warn("minimal_number_of_blocks cannot output a value because the sample is too corrupted. This may impact the following computations.")
                    bloc_nb_min = nan # undefined when blocks are too corrupted
            else:
                b_size_loc_ = KbMOM.block_size_max(n,nb_outliers)
                if b_size_loc_ == 0:
                    bloc_nb_min = nan # undefined when blocks have to be empty
                else:
                    bloc_nb_min = ceil(log(1/alpha) / (2* ((1-outlier_proportion)**b_size_loc_ - 1/2)**2))

        return bloc_nb_min
   
    def stopping_crit(list_all_q_quantile_risks_through_iterations):
        """compute the Aitkens stopping criterion"""
        risk_ = list_all_q_quantile_risks_through_iterations[::-1][:3]
        normalization = (risk_[2]-risk_[1])-(risk_[1]-risk_[0])
        criterion_value = risk_[2] - (risk_[2]-risk_[1])**2/normalization
        return criterion_value
    
    def stopping_crit_GMM(q_quantile_risk):
        """compute the same stopping criterion as in the GMM estimation algorithm"""
        risk_ = q_quantile_risk[::-1][:3]
        Aq = (risk_[0] - risk_[1])/(risk_[1] - risk_[2])
        Rinf = risk_[1] + 1/(1-Aq)*(risk_[0] - risk_[1])
        return Rinf
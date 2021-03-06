'''
hw0.py
Author: Miles Izydorczak
Tufts COMP 135 Intro ML, Fall 2020
Summary
-------
Complete the problems below to demonstrate your mastery of numerical Python.
Submit to the autograder to be evaluated.
You can do a basic check of the doctests via:
$ python -m doctest hw0.py
'''

import numpy as np

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    seed_holder = None
    if random_state is None:
        random_state = np.random 
    elif isinstance(random_state, int):
        seed_holder = random_state
        random_state = np.random.RandomState(random_state)

    L = x_all_LF.shape[0]
    N = int(np.ceil(frac_test * L))

    x_shuffled_LF = random_state.permutation(x_all_LF)

    # Slice the array into the train and test sets
    x_test_NF = x_shuffled_LF[:N]
    x_train_MF = x_shuffled_LF[N:] 

    return x_train_MF, x_test_NF

    ''' Divide provided array into train and test sets along first dimension
    User can provide random number generator object to ensure reproducibility.
    Args
    ----
    x_all_LF : 2D np.array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
        Returned test set will round UP if frac_test * L is not an integer.
        e.g. if L = 10 and frac_test = 0.31, then test set has N=4 examples
    random_state : np.random.RandomState instance or integer or None
        If int, will create RandomState instance with provided value as seed
        If None, defaults to current numpy random number generator np.random.
    Returns
    -------
    x_train_MF : 2D np.array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array
    x_test_NF : 2D np.array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array
    Post Condition
    --------------
    This function should be side-effect free. Provided input array x_all_LF
    should not change at all (not be shuffled, etc.)
    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.201, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True
    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    N = data_NF.shape[0]
    F = data_NF.shape[1]
    Q = query_QF.shape[0]

    # Here we make the result array, to be returned as output of this function
    neighb_QKF = np.zeros((Q, K, F))

    # Iterate through each query and find the K nearest neighbors to that query
    for q in range(Q):
        # Structured array to track the Euclidean distances and their indexes
        dists_idx_N = np.zeros(shape=(N,), dtype=[('distance', '<f8'), ('index', '<i4')])

        for n in range(N):
            differences_sqrd_sum = 0
            for f in range(F):
                # Find the difference between the query's field and the
                # chosen feature vector's field
                difference = query_QF[q][f] - data_NF[n][f]
                # Square that difference
                difference_sqrd = difference ** 2
                # Add the squared distance to the running sum total
                differences_sqrd_sum += difference_sqrd
            # After we have the sum of all of the diffrences squared,
            # take the square root to get the euclidean distance between the
            # data and the query
            euclidean_distance = np.sqrt(differences_sqrd_sum)

            dists_idx_N[n] = (euclidean_distance, n)

        # Now we insert the feature vectors into the result array
        dists_idx_N = np.sort(dists_idx_N, order=['distance', 'index'])
        for k in range(K):
            index_of_neighbor = int(dists_idx_N[k][1])
            neighb_QKF[q][k] = data_NF[index_of_neighbor]
    # Now we should have a complete results array

    return neighb_QKF

    ''' Compute and return k-nearest neighbors under Euclidean distance
    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector
    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    Examples
    --------
    >>> data_NF = np.asarray([
    ...     [1, 0],
    ...     [0, 1],
    ...     [-1, 0],
    ...     [0, -1]])
    >>> query_QF = np.asarray([
    ...     [0.9, 0],
    ...     [0, -0.9]])
    >>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
    >>> neighb_QKF[0]
    array([[1., 0.]])
    >>> neighb_QKF[1]
    array([[ 0., -1.]])
    # Find 3 nearest neighbors
    >>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
    >>> neighb_QKF.shape
    (2, 3, 2)
    >>> neighb_QKF[0]
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., -1.]])
    >>> neighb_QKF[1]
    array([[ 0., -1.],
           [ 1.,  0.],
           [-1.,  0.]])
    '''
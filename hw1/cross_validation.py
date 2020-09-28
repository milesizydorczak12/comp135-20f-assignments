import numpy as np

from performance_metrics import calc_mean_squared_error


def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    Examples
    --------
    # Create simple dataset of N examples where y given x
    # is perfectly explained by a linear regression model
    >>> N = 101
    >>> n_folds = 7
    >>> x_N3 = np.random.RandomState(0).rand(N, 3)
    >>> y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    >>> y_N.shape
    (101,)

    >>> import sklearn.linear_model
    >>> my_regr = sklearn.linear_model.LinearRegression()
    >>> tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(
    ...                 my_regr, x_N3, y_N, n_folds=n_folds, random_state=0)

    # Training error should be indistiguishable from zero
    >>> np.array2string(tr_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'

    # Testing error should be indistinguishable from zero
    >>> np.array2string(te_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'
    '''
    train_error_per_fold = np.zeros(2, dtype=np.int32)
    test_error_per_fold = np.zeros(2, dtype=np.int32)

    # TODO define the folds here by calling your function
    # e.g. ... = make_train_and_test_row_ids_for_n_fold_cv(...)

    # TODO loop over folds and compute the train and test error
    # for the provided estimator

    return train_error_per_fold, test_error_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N

    Examples
    --------
    >>> N = 11
    >>> n_folds = 3
    >>> tr_ids_per_fold, te_ids_per_fold = (
    ...     make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
    >>> len(tr_ids_per_fold)
    3

    # Count of items in training sets
    >>> np.sort([len(tr) for tr in tr_ids_per_fold])
    array([7, 7, 8])

    # Count of items in the test sets
    >>> np.sort([len(te) for te in te_ids_per_fold])
    array([3, 4, 4])

    # Test ids should uniquely cover the interval [0, N)
    >>> np.sort(np.hstack([te_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    # Train ids should cover the interval [0, N) TWICE
    >>> np.sort(np.hstack([tr_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
            8,  9,  9, 10, 10])
    '''
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # TODO obtain a shuffled order of the n_examples

    te_large = int(np.ceil(n_examples / n_folds))
    # print("test_set_size_large:", te_large)
    te_small = te_large - 1
    # print("test_set_size_small:", te_small)


    # pool_of_ints_n = np.arange(0, n_examples)
    # print("pool_of_ints_n", pool_of_ints_n)
    pool_n = random_state.permutation(n_examples)
    # print("pool_n", pool_n)

    train_ids_per_fold = list()
    test_ids_per_fold = list()

    # Determine the size of each fold

    num_folds_large = n_examples % n_folds
    small_folds_start = te_large * num_folds_large

    # The first num_folds_large folds will have the large test set size
    for i in range(num_folds_large):
        start_of_test = i * te_large
        end_of_test = (i + 1) * te_large
        test_ids = pool_n[start_of_test : end_of_test]
        train_ids_before = pool_n[: start_of_test]
        train_ids_after = pool_n[end_of_test :]
        train_ids = np.hstack((train_ids_before, train_ids_after))
        # print("test_ids", test_ids)
        # print("train_ids", train_ids)
        train_ids_per_fold.append(np.sort(train_ids))
        test_ids_per_fold.append(np.sort(test_ids))

    # The remaining folds will be of size 1 smaller than the previous
    for j in range(n_folds - num_folds_large):
        start_of_test = small_folds_start + (j * te_small)
        end_of_test = small_folds_start + ((j + 1) * te_small)
        test_ids = pool_n[start_of_test : end_of_test]
        train_ids_before = pool_n[: start_of_test]
        train_ids_after = pool_n[end_of_test :]
        train_ids = np.hstack((train_ids_before, train_ids_after))
        # print("test_ids", test_ids)
        # print("train_ids", train_ids)
        train_ids_per_fold.append(np.sort(train_ids))
        test_ids_per_fold.append(np.sort(test_ids))        


    # TODO establish the row ids that belong to each fold's
    # train subset and test subset

    return train_ids_per_fold, test_ids_per_fold


x, y = make_train_and_test_row_ids_for_n_fold_cv(7, 4, 9)
print(x)
print(y)


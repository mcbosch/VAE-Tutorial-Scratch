import numpy as np

# ================== ACTIVATION FUNCTIONS

# DONE
class ReLU:
    # DONE
    def __call__(self, x):
        return np.maximum(x,0)

    # DONE
    def partial(self, x):
        return (np.array(x)>0).astype(float) 

    # DONE
    def __str__(self):
        return "ReLU"

# DONE
class Sigmoind:

    # DONE
    def __call__(self, x):
        x = np.clip(x, -500, 500)
        return 1/(1+ np.exp(-x))

    # DONE
    def partial(self, x):
        s = self(x)
        return s*(1-s)
        
    # DONE
    def __str__(self):
        return "Sigmoind"

# DONE
class Softmax:
    def __call__(self, x):
        exps = np.exp(x)  # Maybe we have to add more estability
        if len(x.shape) == 1:
            return exps / np.sum(exps)
        return (exps.T / np.sum(exps, axis=1)).T
    
    def partial(self, x):
         return np.diag(x) - np.outer(x, x)
    
    def __str__(self):
        return "Softmax"


# =================== LOSS FUNCTIONS
    
# DONE
class CrossEntropy:
    """
    The CrossEntropy object is a callable object that computes the CrossEntropy loss of a prediction and it's real value.
    """

    def __call__(self, *args):
        """
        Function that returns the cross entropy between a predicted vector and the real vector.
        The input a batch of different vectors, the input has to be in a matrix in which each row correspond to a vector.
        Parameters
        ----------
            > y_pred: Predicted vector (or batch of vectors)
            > y_true: Real vector (or batch of vectors)
        Returns
        -------
            Returns the CrossEntropy of each vector
        """
        assert len(args) > 0, f"Two parameters expected; just {len(args)} given"
        y_pred, y_true = args[0], args[1]
        # We check the dimensionality (if it comes as )
        if len(y_pred.shape) > 1:
            # We have a batch of vectors; each row is a datapoint
            return -np.sum(y_true * np.log(y_pred + 1e-13), axis = 1)  # evitar log(0)
        else:
            return -np.sum(y_true * np.log(y_pred + 1e-13))

    def partial(self, *args, activated_neurons=True):
    
        assert len(args) > 0, f"Two parameters expected; just {len(args)} given"
        y_pred, y_true = args[0], args[1]
        if activated_neurons: # Computes the partial with respect the activated neurons
            pass
        else: # Computes the partial with respect the non-activated neurons when we use as activation function a Softmax
            return y_pred - y_true
       
    def __str__(self):
        return "CrossEntropy"

# TODO
class SqEuclideanDistance:
    pass

# TODO
class KullbackLeigberg:
    pass

# =================== MANIPULATION OF DATA FUNCTIONS

# TODO - test
def split(data, train_size = 0.8, seed = None):
        if seed != None: 
            np.random.seed(seed)
        
        # We shuffle randomly the data
        np.random.shuffle(data)
        n = len(data)

        index_train = np.random.choice(np.array(range(n)), size=int(np.ceil(n*train_size)), replace=False)
        index_test = np.setdiff1d(np.arange(n), index_train)

        data_train = [data[i] for i in index_train]
        data_test = [data[i] for i in index_test]

        return data_train, data_test

# TODO - test
def random_batches(data, batch_size):
    """
    Randomly partitions data into batches of size batch_size.
    
    Parameters
    ----------
    data : list
        List of observations (any type).
    batch_size : int
        Size of each batch.
    
    Returns
    -------
    batches : list of lists
        Random partition of data into batches.
    """
    data = list(data)  # make sure we can index
    n = len(data)
    
    indices = np.random.permutation(n)
    
    batches = []
    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [data[i] for i in batch_indices]
        batches.append(batch)
    return batches

# TODO - test
def to_one_hot(n, label):
        v = np.zeros(shape = (n,))
        v[label] = 1
        return v

# TODO - test
def to_cov_matrix(log_var, dim_matrix = 1):
    if len(log_var) == 1:
        return np.exp(log_var) * np.eye(dim_matrix) 
    elif len(log_var.shape) == 1:
        return np.diag(np.exp(log_var))
    else:
        raise TypeError("We have not yet implemented a full cov matrix option")
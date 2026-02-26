import numpy as np

# ================== ACTIVATION FUNCTIONS

# DONE - test DONE
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

# DONE - test DONE
class Sigmoid:

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
        return "Sigmoid"

# DONE - test DONE
class Softmax:
    def __call__(self, x):
        x = np.clip(x,-500,500)
        exps = np.exp(x)  # Maybe we have to add more estability
        if len(x.shape) == 1:
            return exps / np.sum(exps)
        return (exps.T / np.sum(exps, axis=1)).T
    
    def partial(self, x):

        x = np.clip(x, -500, 500)
        a = self(x)
        s = x.shape
        if len(s) == 1:
            Id = np.eye(s[0])
            return a[:,np.newaxis]*(Id-a)
        elif len(s) == 2:
            Id = np.array([np.eye(s[1]) for _ in range(s[0])])
            return a[:,:,np.newaxis]*(Id-a[:,np.newaxis,:])
        else:
            raise KeyError("Softmax Partial only supports np.arrays of shape (j,) or (i,j)")
        
        

    def __str__(self):
        return "Softmax"


# =================== LOSS FUNCTIONS
"""
The loss functions allows to receive as an input a matrix. Then it treats each row as an observation and computes the loss for that row. We do this to be able to work with batches. 
"""
# DONE - Test Done
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
        assert len(args) > 0, f"Two parameters expected; {len(args)} given"
        y_pred, y_true = args[0], args[1]
        # We check the dimensionality (if it comes as batch of vectors )
        batch = len(y_pred) if len(y_pred.shape) > 1 else 1
        # We have a batch of vectors; each row is a datapoint
        return -np.sum(y_true * np.log(y_pred + 1e-13))/batch 
        

    def partial(self, *args, activated_neurons=True):
    
        assert len(args) > 0, f"Two parameters expected; just {len(args)} given"
        y_pred, y_true = args[0], args[1]
        batch = len(y_pred) if len(y_pred.shape) > 1 else 1
        if activated_neurons: # Computes the partial with respect the activated neurons
            return - y_true / (batch*(y_pred+1e-13)) # Computationally inestable
        else: # Computes the partial with respect the non-activated neurons when we use as activation function a Softmax
            return (y_pred - y_true)/batch
       
    def __str__(self):
        return "CrossEntropy"

# DONE - test DONE
class MSE:

    def __init__(self):
        self.valid_respect_to = ["x","y"]

    def __call__(self, *args):
        X, Y = args[0], args[1]
        batch = len(X) if len(X.shape) > 1 else 1
        dif = (X - Y)**2
        return 0.5*np.sum(dif)/batch
        
    def partial(self, *args, respect_to = "x"):
        if respect_to not in self.valid_respect_to:
            raise KeyError(f"Please introduce a valid input for 'respect_to'\nOptions: {self.valid_respect_to}")
        
        x, y = args[0], args[1]
        batch = len(x) if len(x.shape) > 1 else 1
        if respect_to == "y":
            return (y-x)/batch
        elif respect_to == "x":
            return (x-y)/batch

    def __str__(self):
        return "MSE"

# DONE TODO - test (for VAE)
class KullbackLeibler:
    """
    Computes the Kullback-Leibler Divergence of two normals.
    """
    def __init__(self):
        self.G = ["Diagonal", "Isotropic", "FullCov"]
        self.partials_options = ["mean", "logvar"]
    def __call__(self, *args, Gaussian_Variance = "Diagonal"):
        if Gaussian_Variance not in self.G:
            raise TypeError(f"Please introduce a valid parameter for Gaussian_Variance. \nPossible parameters: {self.G}")
        if Gaussian_Variance == "Diagonal":
            mu, logvar = args[0], args[1]
            return 0.5*(np.sum(np.exp(logvar)**2) + np.sum(mu**2) - len(mu) - 2*np.sum(logvar))
        elif Gaussian_Variance == "Isotropic":
            pass
        elif Gaussian_Variance == "FullCov":
            pass

    def partial(self, *args,  respect = "mean", Gaussian_Variance = "Diagonal"):
        if Gaussian_Variance not in self.G:
            raise TypeError(f"Please introduce a valid parameter for Gaussian_Variance. \nPossible parameters: {self.G}")
        if respect not in self.partials_options:
            raise TypeError(f"Please introduce a valid parameter for respect. \nPossible parameters: {self.partials_options}")
        if Gaussian_Variance == "Diagonal":
            meanX, logvarX = args[0], args[1]
            if respect == "mean":
                return meanX
            if respect == "logvar":
                return np.exp(logvarX)**2 - np.ones(shape=logvarX.shape)
            
    def __str__(self):
        return "KullbackLeibler"

# =================== MANIPULATION OF DATA FUNCTIONS

# TODO - test
def split(data, train_size = 0.8, seed = None):
        # Extend for pandas data
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
        batch_x, batch_y = [], []
        for i in batch_indices:
            batch_x.append(data[i][0])
            batch_y.append(data[i][1])
        batch = (np.array(batch_x), np.array(batch_y))
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
    

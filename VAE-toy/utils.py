import numpy as np


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

class Softmax:
    def __call__(self, x):
        exps = np.exp(x - np.max(x))  # numéricamente estable
        return exps / np.sum(exps)
    
    def partial(self, x):
         return np.diag(x) - np.outer(x, x)
    
    def __str__(self):
        return "Softmax"

class CrossEntropy:
    def __call__(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-12))  # evitar log(0)
    
    def partial(self, y_pred, y_true):
        return y_pred - y_true  # si se usa con softmax
       
    def __str__(self):
        return "CrossEntropy"

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

def to_one_hot(n, label):
        v = np.zeros(shape = (n,))
        v[label] = 1
        return v

def to_cov_matrix(log_var, dim_matrix = 1):
    if len(log_var) == 1:
        return np.exp(log_var) * np.eye(dim_matrix) 
    elif len(log_var.shape) == 1:
        return np.diag(np.exp(log_var))
    else:
        raise TypeError("We have not yet implemented a full cov matrix option")
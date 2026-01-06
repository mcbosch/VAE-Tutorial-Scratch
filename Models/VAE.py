import numpy as np
from utils import *
from NN import *
"""
If the reader doesn't have good knowledge of Variational AutoEncoders we recomend reeding all the comments. It has a brew eplanation. Also you can read the README.md file for detailed information. 
"""
class VAE:
    """
    This class implements a basic AutoEncoder. It consists of an encoder and a decoder, with a latent space in between.
    This class aims to be flexible, so you can use it in different contexts and be easily modifiable. 

    ===== FORMAT OF INPUTS
    We define a easy format of inputs. 
        (1) numpy.array with constant shape (d,)
        (2) list with first element a python object of type (1)
    Dataset: D = list of objects (1) or (2)

    ===== EXAMPLE OF USSAGE
        >>> np.random.seed(123)
        >>> ae = AutoEncoder(input_size = 786, 
                            hidden_layers = 2, 
                            dimensions = [128,128],
                            latentsdim = 20)
        >>> # We have a dataset in a variable D
        >>> print(ae)
            AutoEncoder Object.
            input:      x (np.arrray(786,)) 
            forward:    x -> encoder -> latent.space(dim=20) -> decoder -> x'
            output:     x' (np.array(786,))
            encoder: x --> z (latent space)
                epsilon <- N(0,I(l)) # Gaussian Noise
                NeuralNetwork: outputs (mu, log(sigma))
                    (786,) => LL + ReLU => (128,)
                    (128,) => LL + ReLU => (128,)
                    (128,) => LL + ReLU => (40,)
                z <- mu + epsilon · sigma (· is pointwise product) 
            decoder: z --> x'
                NeuralNetwork
                    (20,) => LL + ReLU => (128,)
                    (128,) => LL + ReLU => (128,)
                    (128,) => LL + Sigmoind => (786,)
        >>> ae.train(D) 
        >>> ae.forward(D[0]) # returns a numpy.array of shape (786,)
            [ 0.10 ... 0]
        >>> ae.generate() # returns a numpy.array of shape (786,)
            [ 0.12 ... 0.7] 

    """
    # TODO
    def __init__(self, 
                 n_input, 
                 n_hidden_1, 
                 n_hidden_2,
                 dimensions_1 = [], 
                 mu_activation = "ReLU",
                 logvar_activation = "ReLU",
                 activations_1 = [],
                 dimensions_2 = [],
                 activations_2 = [], 
                 latent_dim = 10):
        """
        Parameters:
            · n_input: dimension of the input
            · n_hidden: dimension of the hidden layers in the encoder and decoder NNs
        Returns:
            · An AutoEncoder object
        """
        self.encoder = Sequence([])
        self.decoder = Sequence([])

        if activations_1 == []:
            print("\033[1;31mWARNING\033[0m - There are no activation functions speciefieds in encoder NN, we use ReLU")
            activations_1 = ['ReLU' for _ in range(n_hidden_1)]
        
        for i in range(n_hidden_1-1):
            if i == 0:
                self.encoder.add(LinearLayer(
                         n_input, dimensions_1[i], activation = activations_1[i]
                    ))
            else:
                self.encoder.add(LinearLayer(
                         dimensions_1[i-1],dimensions_1[i], activation = activations_1[i]
                    ))
        self.mu = LinearLayer(dimensions_1[-1], latent_dim, activation = mu_activation)
        self.logvar = LinearLayer(dimensions_1[-1], latent_dim, activation = logvar_activation)

        if activations_2 == []:
            print("\033[1;31mWARNING\033[0m - There are no activation functions speciefieds in decoder NN, we use ReLU")
            activations_2 = ['ReLU' for _ in range(n_hidden_2)]

        for i in range(n_hidden_2):
            if i == 0:
                self.encoder.add(LinearLayer(
                         latent_dim, dimensions_2[i], activation = activations_2[i]
                    ))
            else:
                self.encoder.add(LinearLayer(
                         dimensions_2[i-1], dimensions_2[i], activation = activations_2[i]
                    ))

        self.latent_dim = latent_dim
        self.latent_vars = np.zeros(shape = (latent_dim, ))
        self.epsilon = np.zeros(shape = (latent_dim, ))

    # DONE
    def forward(self, x):
            mu, log_sigma = self.encoder.forward(x)
            Sigma = to_cov_matrix(log_sigma)
            
            self.epsilon = np.random.multivariate_normal(np.zeros(shape = (self.latent_dim)),np.eye(self.latent_dim))
            z = mu + Sigma @ self.epsilon
            
            return self.decoder.forward(z)
    
    # TODO
    def loss(self, x):
        pass
    
    # TODO
    def train(self, D, step = 0.01, batch = 10, epochs = 100):
         pass

    # TODO
    def __str__(self):
            pass

    # TODO
    def description(self):
        raise TypeError("\033[1;31mFUNCTION NOT IMPLEMENTED\033[0m")


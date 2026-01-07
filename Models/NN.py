import numpy as np
from utils import *

"""
This script defines a general NeuralNetwork class. We do it in a general way so we have a flexible model with an easy way of tracking errors. See the README file to more information of how this class works.

Example of ussage: 
"""

class NeuralNetwork:

    # Define which Loss and Activations functions supports our Neural Network
    LossFunctions = ["CrossEntropy"]
    ActivationFunctions = ["ReLU", "Sigmoind", "Softmax"]
    
    AF = {"ReLU": ReLU(), "Sigmoind": Sigmoind(), "Softmax": Softmax()}

    # DONE
    def __init__(self, n_in, n_out, n_layers, dimensions, activations):
        """
        Parameters
        ----------
            · n_in: dimension of the input
            · n_out: dimension of the output
            · n_layers: number of hidden layers
            · dimensions: list of the dimension of each hidden layer
            · activations: list of the activation function that must be applied after each hidden layer.
        """

        if n_layers != len(dimensions): 
            raise TypeError("The length of dimensions must be the number of hidden layers")
        
        if n_layers != len(activations): 
            raise TypeError("The length of activations must be the number of hidden layers")
        
        self.n_layers = n_layers
        self.n_in = n_in
        self.n_out = n_out

        self.layers = []
        self.dimensions = dimensions
        self.activations = activations

        if self.n_layers == 0: # We don't have hidden layers
            self.layers.append(LinearLayer(self.n_in, self.n_out, activation=Softmax()))

        else:
            for i in range(n_layers):
                if activations[i] not in NeuralNetwork.ActivationFunctions:
                        raise TypeError(f"Please, use some of the \033[1;31mavailable activation funcions: {NeuralNetwork.ActivationFunctions}\033[0m")
                
                if i == 0:
                    # Create the first layer
                    self.layers.append(LinearLayer(self.n_in, dimensions[i], activation=activations[i]))
                else:
                    # Create hidden layers
                    self.layers.append(LinearLayer(dimensions[i-1], dimensions[i], activation=activations[i]))
        
            # Create last layer
            self.layers.append(LinearLayer(self.dimensions[-1], self.n_out, activation= "Softmax"))

        self.n_trainable_parameters = 0
        for i in range(n_layers): self.n_trainable_parameters += self.layers[i].n_trainable_parameters
    
    # DONE
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input

    # DONE
    def train(self, data, epochs, learning_rate=0.1, loss=CrossEntropy()):
        """
        Parameters
        ----------
            · data: data must be a list with touples of (np.array(size=(n,)), label), where the labels should be a number between 0 and the number of possible labels minus 1.
            · epochs: the number of iterations that we apply the backpropagation.
            · learning_rate: stepsize of the backpropagation algorithm.
            · loss: the loss functions that we use to train the model.
        
        Returns
        -------
            Returns the same neural networks with the parameters updated. 
        """

        if str(loss) not in NeuralNetwork.LossFunctions: 
            raise TypeError(f"Please use some of the \033[1;31mavailable loss functions: {NeuralNetwork.LossFunctions}\033[0m")

        print(f"\033[1;33mTrainable Parameters:\033[0m\t{self.n_trainable_parameters}")
        print(f"\033[1;32mNumber of Epochs:\033[0m\t{epochs}\n")
        
        for _ in range(epochs):
            for x, y  in data:
                pred = self.forward(x)
                y = self.to_one_hot(y)
                
                # d values represent the backpropagation errors
                d = pred-y
                self.layers[-1].deltas = d
                d = self.layers[-1].weights.T @ d

                # Backpropagate errors
                for i in range(self.n_layers):
                        l = self.layers[self.n_layers-1-i]
                        d = l.backpropagation(d)
                
                # Update weights
                for l in self.layers: l.update_parameters(learning_rate)

    # DONE
    def test(self, data):
        """
        Parameters
        ----------
            · data: data to test the model
        Returns
        -------
            · acc: the accuracy of the model for the data.
        """
        n = len(data)
        acc = 0

        for i in range(n):
            y = data[i][1]
            pred = np.argmax(self.forward(data[i][0]))
            acc +=  pred == y
        return acc/n
    
    # DONE
    def to_one_hot(self, label):
        v = np.zeros(shape = (self.n_out,))
        v[label] = 1
        return v

    # DONE
    def __str__(self):
        s = "Neural Network\n"
        ml = len(s)
        s += '-'*(len(s)-1) + '\n'
        
        s += f'>\033[1;32m Number of Hidden Layers: \033[0m ({self.n_layers})\n'
        s += f'>\033[1;33m Pass of Information: \033[0m\n'
        for l in self.layers:
            s += '\t'+ str(l) + '\n'
        s += "-"*ml
        return s
    
        
class LinearLayer:

    # DONE
    def __init__(self,
                 n_cels_in,
                 n_cels_out, 
                 bias = True, 
                 activation = 'ReLU',
                 batch = 1):
        """
        Parameters
        ----------
            · n_cels_in:
            · n_cels_out:
            · bias:
            · activation: defines wich activation layer we want to apply at the output.
        """

        if activation not in NeuralNetwork.ActivationFunctions:
            raise TypeError(f"Please, use some of the \033[1;31mavailable activation funcions: {NeuralNetwork.ActivationFunctions}\033[0m")
        
        self.n_in = n_cels_in
        self.n_out = n_cels_out
        self.n_trainable_parameters = self.n_out*self.n_in + (self.n_out if bias else 0)
        
        # Initialize weights randomly and normalize
        limit = np.sqrt(1 / self.n_in)
        self.weights = np.random.randn(self.n_in, self.n_out) * limit
        self.bias = np.random.randn(self.n_out) * limit

        self.activation = NeuralNetwork.AF[activation]
        
        # Values of the neurons
        self.x = np.zeros(shape=(self.n_in,))
        self.z = np.zeros(shape=(self.n_out,))
        self.a = np.zeros(shape=(self.n_out,))
        self.cache = np.zeros(shape=(self.n_out, batch)) # The error of a Neuron
    
    # DONE
    def forward(self, input):
        # Update values of the neurons
        self.x = input
        self.z = input @ self.weights + self.bias 
        self.a = self.activation(self.z)
        return self.a
    
    # DONE
    def backpropagation(self, d):
        self.cache =  d * self.activation.partial(self.z)
        return self.cache @ self.weights.T

    def update_parameters(self, learning_rate):
        # Verify if we are working by batches
        self.weights = self.weights - learning_rate*(self.x.T @ self.cache)
        if len(self.cache.shape) == 1:
            self.bias = self.bias - learning_rate* self.cache
        else:
            self.bias = self.bias - learning_rate* np.sum(self.cache, axis=0)
    # DONE
    def __str__(self):
        s = f"dim(\033[1;33m{self.n_in}\033[0m) -- Fully Conected --> dim(\033[1;33m{self.n_out}\033[0m) --> \033[1;32m{str(self.activation)}\033[0m"
        return s


class Sequence:
    def __init__(self, layers):
        # Verify that the dimensions are correct
        n = len(layers)
        for i in range(n-1):
            if layers[i].n_out != layers[i+1].n_in:
                raise ValueError("Incompatible layer dimensions")
        
        self.layers = layers
        self.n_layers = n
        self.size_in = layers[0].n_in
        self.size_out = layers[-1].n_out

    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input

    def add(self, layer):
        self.layers.append(layer)
    
    def backpropagate(self, step, delta, update_parameters = False):
        for layer in self.layers:
            delta = layer.backpropagate(delta)
            if update_parameters:
                layer.update_parameters(learning_rate = step)
            

        
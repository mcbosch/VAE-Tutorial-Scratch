import numpy as np  
import time
import sys

from nn.utils import *

"""
This script defines a general NeuralNetwork class. We do it in a general way so we have a flexible model with an easy way of tracking errors. See the README file to more information of how this class works.

Example of ussage: 
"""
# TODO - test
class NeuralNetwork:

    # Define which Loss and Activations functions supports our Neural Network
    LossFunctions = ["CrossEntropy", "MSE"]
    ActivationFunctions = ["ReLU", "Sigmoid", "Softmax"]
    
    AF = {"ReLU": ReLU(), "Sigmoid": Sigmoid(), "Softmax": Softmax()}

    # DONE TODO - test
    def __init__(self, 
                 layers):
        
        self.name = "NN"
        self.layers = Sequence([])
        last_dim = -1

        for layer in layers:
            if last_dim < 0: last_dim = layer.n_out
            else: 
                assert layer.n_in == last_dim, "Check the layers have correct dimensions"
                last_dim = layer.n_out
            self.layers.add(layer)
            # Create last layer

        self.n_trainable_parameters = 0
        for i in range(len(self.layers)): self.n_trainable_parameters += self.layers[i].n_trainable_parameters
    
    def set_name(self,name):
        self.name = name

    def forward(self, input):
        return self.layers.forward(input)

    def train(self, 
              data, 
              epochs, 
              learning_rate = 0.01,
              batch_size = 1, 
              beta_1 = 0.9,
              beta_2 = 0.99, 
              loss = CrossEntropy(), 
              adam = True,
              data_val = None):
        """
        IMPORTANT
        ---------
            data format:
        """

        if str(loss) not in NeuralNetwork.LossFunctions: 
            raise TypeError(f"Please use some of the \033[1;31mavailable loss functions: {NeuralNetwork.LossFunctions}\033[0m")
        
        print(f"\033[1m" + "="*8 + f"\tTraining {self.name}\t" + "="*8)
        print(f"\033[1;34mDatapoints:\033[0m\t{len(data)}\tGrouped in batches of size {batch_size}")
        print(f"\033[1;33mTrainable Parameters:\033[0m\t{self.n_trainable_parameters}")
        print(f"\033[1;32mNumber of Epochs:\033[0m\t{epochs}\n")
        
        total = len(data)
        bar_width = 40
        results = {'Epochs': {'loss_train': [], 'loss_val': [], 'acc_val': []}, 'Time': None}

        start_training_model = time.time()
        sys.stdout.write(f"\033[1mTraining:\t{'':40s}\n")
        # ----- Initialize Momentums
        momentums = [(0,0) for _ in range(len(self.layers))] if adam else None
        t = 0
        for e in range(epochs):
            loss_epoch = []
            
            # Suhhle and batch data
            data_loader = random_batches(data, batch_size)
            total = len(data_loader)

            # ----- Print format training
            p1 = int(bar_width * (e+1) / epochs)
            perc1 = int(100*(e)/epochs)
            sys.stdout.write("\033[1A")
            sys.stdout.write(f"\r\033[1mTraining {self.name}:\t|\033[31;7m{' '*p1}\033[0m{' '*(bar_width-p1)}| {perc1:02d}%\n")
            sys.stdout.flush()
            sys.stdout.write(f"\r\tEpoch: {e+1:02d}\t|{'':40s}|")
            sys.stdout.flush()
            
            # ----- Run over data
            i = 0
            for X in data_loader:
                i+=1
                x, y = X[0], X[1] 
               
                # ------ Print format training
                p2 = int(bar_width * i / total)
                perc2 = int(100*i/total)
                sys.stdout.write(f"\r\tEpoch {e+1:02d}:\t|\033[32;7m{' '*p2}\033[0m{' '*(bar_width-p2)}| {perc2:02d}%")
                sys.stdout.flush()

                # ------ Forward datapoint
                pred = self.forward(x)
                y = self.to_one_hot_batch(y)

                # ----- Compute loss and partial error
                loss_epoch.append(loss(pred,y))
                first_delta_comp = str(loss) == "CrossEntropy" and str(self.layers[-1].activation) == "Softmax"
                delta = loss.partial(pred,y,activated_neurons= not first_delta_comp) if first_delta_comp else loss.partial(pred,y)

                # ------ Backpropagate error
                current_batch = len(x)
                t+=1
                _, momentums = self.layers.backpropagate(step=learning_rate,
                                               beta_1 = beta_1,
                                               beta_2 = beta_2,
                                               momentums = momentums,
                                               delta = delta,
                                               batch = current_batch,
                                               t=t,
                                               update_parameters=True, 
                                               first_delta_computed= first_delta_comp, 
                                               adam=adam)
                
            # ------ Epoch finished: Save result
            results["Epochs"]["loss_train"].append(np.mean(loss_epoch))
            if data_val != None:
                acc, ls = self.test(data_val, loss)
                results["Epochs"]["loss_val"].append(ls)
                results["Epochs"]["acc_val"].append(acc)
            
        # ------ Finish training
        results["Time"] = time.time() - start_training_model
        return results
               
    def test(self, data, loss):
        """
        Parameters
        ----------
            · data: data to test the model
        Returns
        -------
            · acc: the accuracy of the model for the data.
            · loss: loss of the model for the data
        """
        n = len(data)
        acc = 0
        ls = 0
        
        for i in range(n):
            y = data[i][1]
            y_hat = self.forward(data[i][0])
            pred = np.argmax(y_hat)
            ls += loss(y_hat, self.to_one_hot(y))
            acc +=  pred == y
        return acc/n, ls/n
    
    def to_one_hot(self, label):
        n = self.layers[-1].n_out
        v = np.zeros(shape = (n,))
        v[label] = 1
        return v
    
    def to_one_hot_batch(self, labels):
        n = self.layers[-1].n_out
        batch = len(labels)
        Y = np.zeros((batch, n))
        Y[np.arange(batch), labels] = 1
        return Y


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

    def __init__(self,
                 n_cels_in,
                 n_cels_out, 
                 bias = True, 
                 activation = 'ReLU',
                 batch = 1,
                 random_seed = None):
        """
        Parameters
        ----------
            · n_cels_in:
            · n_cels_out:
            · bias:
            · activation: defines wich activation layer we want to apply at the output.
        """
        if random_seed != None:
            np.random.seed(random_seed)

        if activation not in NeuralNetwork.ActivationFunctions:
            raise TypeError(f"Please, use some of the \033[1;31mavailable activation funcions: {NeuralNetwork.ActivationFunctions}\033[0m")
        
        self.n_in = n_cels_in
        self.n_out = n_cels_out
        self.n_trainable_parameters = self.n_out*self.n_in + (self.n_out if bias else 0)
        
        # Initialize weights randomly and normalize
        limit = np.sqrt(2.0 / self.n_in)
        self.weights = np.random.randn(self.n_in, self.n_out) * limit
        self.bias = np.zeros(self.n_out)

        self.activation = NeuralNetwork.AF[activation]
        
        # Values of the neurons
        self.x = None
        self.z = None
        self.cache = None 
    
    def forward(self, input):
        # Update values of the neurons
        self.x = input # shape: (batch, n_in) 
        self.z = input @ self.weights + self.bias 
        return self.activation(self.z)
        
    def backpropagation(self, d, first_delta_computed = False):
   
        if first_delta_computed:
            self.cache = np.array(d)
        elif str(self.activation) == 'Softmax':
            self.cache = np.vecmat(d,self.activation.partial(self.z))
        else:
            self.cache = d*self.activation.partial(self.z)
        
        return self.cache @ self.weights.T

    def update_parameters(self, 
                          learning_rate, 
                          beta_1 = 0.9, 
                          beta_2 = 0.99, 
                          m0 = 0, 
                          v0 = 0, 
                          batch = 1,
                          t = 1, 
                          adam = True):
        # Verify if we are working by batches and compute partial of weights in each case

        if batch == 1:
            partial_bias = self.cache
            partial_weights = np.outer(self.x, self.cache)
        else:
            partial_bias = np.sum(self.cache, axis=0)
            partial_weights = np.sum(self.x[:,:,np.newaxis] @ self.cache[:,np.newaxis,:], axis=0)
        
        if adam:
            # Compute first momentum
            m1 = beta_1 * m0 + (1 - beta_1) * partial_weights
            # Compute second momentum
            v1 = beta_2 * v0 + (1 - beta_2) * partial_weights**2
            # Bias correction
            m1h = m1/(1-beta_1**t)
            v1h = v1/(1-beta_2**t)
            # Update parameters
            self.weights -= (learning_rate / (np.sqrt(v1h) + 1e-8)) * m1h
            self.bias -= learning_rate * partial_bias
            return m1, v1
        
        else:
            self.weights -= learning_rate * partial_weights
            self.bias -= learning_rate * partial_bias
            return None, None
        
    def __str__(self):
        s = f"dim(\033[1;33m{self.n_in}\033[0m) -- Fully Conected --> dim(\033[1;33m{self.n_out}\033[0m) --> \033[1;32m{str(self.activation)}\033[0m"
        return s

# DONE
class Sequence:

    # DONE
    def __init__(self, layers):
        # Verify that the dimensions are correct
        n = len(layers)
        for i in range(n-1):
            if layers[i].n_out != layers[i+1].n_in:
                raise ValueError("Incompatible layer dimensions")
        
        self.layers = layers
        self.n_layers = n
        self.size_in = layers[0].n_in if n != 0 else 0
        self.size_out = layers[-1].n_out if n != 0 else 0

    # DONE
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input

    # DONE 
    def add(self, layer):
        self.layers.append(layer)
    
    def __getitem__(self,index):
        return self.layers[index]
    
    def __len__(self):
        return len(self.layers)
    
    # DONE
    def backpropagate(self, 
                      delta, 
                      step=0.1, 
                      beta_1 = 0.9,
                      beta_2 = 0.9, 
                      momentums = None,
                      batch = 1,
                      t = 1,
                      update_parameters = False, 
                      first_delta_computed = False, 
                      adam=True):

        
        n = len(self.layers)
        for i in range(n):
            layer_idx = n - 1 - i
            layer = self.layers[layer_idx]
            if i == 0:
                delta = layer.backpropagation(delta, first_delta_computed = first_delta_computed)
            else:
                delta = layer.backpropagation(delta)

            if update_parameters:
                if momentums is not None and adam:
                    m0, v0 = momentums[layer_idx]
                    m1, v1 = layer.update_parameters(learning_rate = step,
                                         beta_1 = beta_1, 
                                         beta_2 = beta_2, 
                                         m0 = m0, 
                                         v0 = v0, 
                                         batch = batch,
                                         t = t,
                                         adam=adam)
                    momentums[layer_idx] = (m1,v1)
                else:
                    layer.update_parameters(learning_rate = step, 
                                            batch = batch)

            
        return delta, momentums
            

        
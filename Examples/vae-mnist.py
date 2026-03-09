from nn.VAE import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train shape: (60000, 28, 28)
# Convert to float and normalize
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# Flatten images to vectors
x_train = x_train.reshape(len(x_train), -1)  # (60000, 784)
x_test  = x_test.reshape(len(x_test), -1)    # (10000, 784)

model = VAE(784, 
            n_hidden_1 = 2, 
            n_hidden_2 = 2, 
            dimensions_1 = [524, 124],
            mu_activation = "ReLU",
            logvar_activation = "ReLU",
            activations_1 = ["ReLU", "ReLU"],
            dimensions_2 = [124, 524],
            activations_2 = ["ReLU", "ReLU", "Sigmoind"],
            latent_dim = 124)

mean_loss, results = model.train(x_train)

plt.plot(list(range(100)), mean_loss)
plt.show()

# Run loss over test


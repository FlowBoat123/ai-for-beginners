import numpy as np 
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from fc_layer import FCLayer
from convolutional_layer import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import predict, train

def process_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indicies = np.hstack((zero_index, one_index))
    all_indicies = np.random.permutation(all_indicies)
    x, y = x[all_indicies], y[all_indicies]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# Load MNIST from server    
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process_data(x_train, y_train, 100)
x_test, y_test = process_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    FCLayer(5 * 26 * 26, 100),
    Sigmoid(),
    FCLayer(100, 2),
    Sigmoid()
]
# training
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
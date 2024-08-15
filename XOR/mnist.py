import numpy as np 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from network import Network 
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation import tanh, tanh_prime
from losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


(x_train, y_train) ,(x_test, y_test) = mnist.load_data()

# training data: 6000 samples
# reshape and normalize: input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = to_categorical(y_train)

# test data: 1000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# Network
net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100,50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50,10))
net.add(ActivationLayer(tanh,tanh_prime))

# train on 1000 samples
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs = 35, learning_rate = 0.1)

out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
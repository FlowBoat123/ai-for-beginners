import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

np.random.seed(13)

train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

input_dim = 1
output_dim = 1
learning_rate = 0.1

# weight matrix
w = tf.Variable([[100.0]])
# bias vector
b = tf.Variable(tf.zeros(shape=(output_dim,)))

def f(x):
    return tf.matmul(x,w) + b

def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))
@tf.function
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = f(x)
        loss = compute_loss(y, predictions)
        # gradient
        dloss_dw, dloss_db = tape.gradient(loss, [w,b])
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss

#shuffle the data
indicies = np.random.permutation(len(train_x))
features = tf.constant(train_x[indicies], dtype=tf.float32)
labels = tf.constant(train_labels[indicies], dtype=tf.float32)

batch_size = 4
for epoch in range(10):
    for i in range(0, len(features),batch_size):
        loss = train_on_batch(tf.reshape(features[i:i+batch_size],(-1,1)),tf.reshape(labels[i:i+batch_size],(-1,1)))
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

plt.scatter(train_x, train_labels)
x = np.array([min(train_x), max(train_x)])
y = w.numpy()[0,0]*x+b.numpy()[0]
plt.plot(x,y,color='red')
plt.show()
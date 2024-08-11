import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow  as tf 
import numpy as np 

a = tf.constant([[1,2], [3,4]])
a = tf.random.normal(shape=(10,3))
print(a)
s = tf.Variable(tf.zeros_like(a[0]))
for i in a:
  s.assign_add(i)

print(s)
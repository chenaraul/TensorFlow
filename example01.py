import tensorflow as tf
import numpy as np

#create a phony data
x_data = np.float32(np.random.rand(2,100)) #random input
y_data = np.dot([0.100,0.200],x_data) + 0.300

#build a linear model
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1,1))
y = tf.matmul(W,x_data) + b

#minimize variance
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#initialize variable
init = tf.global_variables_initializer()

#launch map
sess = tf.Session()
sess.run(init)

#fit plane
for step in xrange(0,201):
    sess.run(train)
    if step % 20 == 0:
        print step , sess.run(W) , sess.run(b)
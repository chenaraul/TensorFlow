#01 op
"""
import tensorflow as tf

#create a constant op,generate a 1*2 matrix, as a node
matrix01 = tf.constant([[3.0,3.0]])

#create another constant op,generate a 2*1 matrix
matrix02 = tf.constant([[2.0],[2.0]])

#create a matirx multiplication
product = tf.matmul(matrix01,matrix02)

#start dafault graph
sess = tf.Session()

result = sess.run(product)
print result

sess.close()


#02 with 

import tensorflow as tf

with tf.Session() as sess:
    with tf.device('/cpu:0):
        matrix01 = tf.constant([[3.0,3.0]])
        matrix02 = tf.constant([[2.0],[2.0]])
        product = tf.matmul(matrix01,matrix02)
        print sess.run(product)


#03 enter an interactive session
import tensorflow as tf

sess =tf.InteractiveSession()

x = tf.Variable([2.0,3.0])
y = tf.constant([1.0,1.5])

x.initializer.run()

sub01 = tf.subtract(x, y)
print sub01.eval()



#04 variables

import tensorflow as tf

state = tf.Variable(1,name="counter")
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run(init_op)
        print sess.run(state)
        for _ in range(3):
            sess.run(update)
            print sess.run(state)

"""

#05 fetch
import tensorflow as tf

input01 = tf.constant(3.0)
input02 = tf.constant(2.0)
input03 = tf.constant(4.0)

input04 = tf.placeholder(tf.float32)
input05 = tf.placeholder(tf.float32)

add01 = tf.add(input01,input02)
mul01 = tf.multiply(input03,add01)

mul02 = tf.multiply(input04,input05)

with tf.Session() as sess:
    print sess.run([mul01,add01])
    print sess.run([mul02],feed_dict={input04:[7.],input05:[8.]})






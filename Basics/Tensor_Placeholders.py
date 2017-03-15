import tensorflow as tf

# construct a graph with placeholders
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
c = a + b
# '+' for tf.add()

# Tensors' shape is not defined
# so it can accept any shape of tensor
# placeholder accept value by using feed_dict parameter

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    scalar_c = sess.run(c, feed_dict={a:3, b:4})
    vector_c = sess.run(c, feed_dict={a:[3,4], b:[5,6]})
    tensor_c = sess.run(c, feed_dict={a:[[1,2],[3,4]], b:[[5,6],[7,8]]})
    print('Scalar:\n', scalar_c)
    print('Vector:\n', vector_c)
    print('Tensor:\n', tensor_c)

import tensorflow as tf

# construct a simple linear model with Variables
x = tf.placeholder(tf.float32)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Tensorflow Variables can be initialized when tf.Variable() is called
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
# be you can change the the value by using tf.assign()

linear_model = W * x + b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Variables need to be initialized before evaluating the graph
    model_output = sess.run(linear_model, feed_dict={x:[1,2,3,4]})
    print(model_output)
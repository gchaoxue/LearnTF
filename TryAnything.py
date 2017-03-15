import tensorflow as tf

x = tf.Variable([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]], dtype=tf.float32)
sum_x = tf.reduce_sum(x, axis=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(sum_x)
    print(sum_x.eval())
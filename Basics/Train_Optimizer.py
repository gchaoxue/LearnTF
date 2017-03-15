import tensorflow as tf

# calculate gradient automatically by using tf.gradients()

# train a linear model
x = tf.placeholder(tf.float32)
W = tf.Variable([-.1], tf.float32)
b = tf.Variable([0.], tf.float32)
linear_model = W * x + b

# define the placeholder for the desired values
y = tf.placeholder(tf.float32)
# define the loss function
square_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(square_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    # expecting W->-1, b->1

    print(sess.run([W, b]))
    # [array([-0.99999768], dtype=float32), array([ 0.99999309], dtype=float32)]
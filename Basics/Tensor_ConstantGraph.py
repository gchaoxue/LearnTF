import tensorflow as tf

# construct a graph with constant tensors
node1 = tf.constant(value=3.0, dtype=tf.float32)
node2 = tf.constant(value=4.0)
node3 = tf.add(node1, node2)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # tf.constant is not a Tensorflow Variable so it don't need a initialization

    print('Tensor node1:', node1)
    print('Tensor node2:', node2)
    print('Tensor node3:', node3)
    # Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
    # showing the tensor's type, shape and dataType

    n1 = sess.run(node1)
    n2 = sess.run(node2)
    n3 = sess.run(node3)
    # the tensor value is evaluated with Tensorflow Session
    print('Tensor value node1:', n1)
    print('Tensor value node2:', n2)
    print('Tensor value node3:', n3)
    # now you can print out the value of tensors

import tensorflow as tf

# The difference between tf.get_variable() and tf.Variable()
# http://stackoverflow.com/questions/35919020
#
# tf.get_variable() can be used with the name of the variable as argument
# to either create a new variable with such name or retrieve the one that
# was created before.
#
# tf.Variable() will create a new variable every time it is called
# and it will add a suffix(0 based id) to the variable name if a variable
# with such name already exists.

# tf.variable_scope returns a context manager for defining ops that creates variables.
# tf.variable_scope will implicitly opens a tf.name_scope which only add prefix to variables that
# created by tf.Variable.

with tf.name_scope('foo'):
    var_1 = tf.get_variable(name='var_1', shape=[1], dtype=tf.float32)
    var_2 = tf.Variable(name='var_2', initial_value=[1], dtype=tf.float32)
    sum = var_1 + var_2
    print(var_1.name)
    # var_1:0
    print(var_2.name)
    # foo/var_2:0
    print(sum.name)
    # foo/add:0

with tf.variable_scope('bar'):
    var_1 = tf.get_variable(name='var_1', shape=[1], dtype=tf.float32)
    var_2 = tf.Variable(name='var_2', initial_value=[1], dtype=tf.float32)
    sum = var_1 + var_2
    print(var_1.name)
    # bar/var_1:0
    print(var_2.name)
    # bar/var_2:0
    print(sum.name)
    # bar/add:0

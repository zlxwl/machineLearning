import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print(input_data)

mnist = input_data.read_data_sets('miniset_data', one_hot=True)

batch_size = 128

# 多少个批次
n_batch = mnist.train.num_examples // batch_size


def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x_input')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y_input')

# 创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('weight'):
        W_1 = tf.Variable(tf.zeros([784, 10]), name='W')
    with tf.name_scope('bias'):
        bias_1 = tf.Variable(tf.zeros([10]), name='b')
    with tf.name_scope('wx_plus_b'):
        Weights_1 = tf.matmul(x, W_1) + bias_1
    with tf.name_scope('softmaxt'):
        prediction = tf.nn.softmax(Weights_1)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

with tf.name_scope('train_batch'):
    train_batch = tf.train.AdagradOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([train_batch], feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(epoch, acc)
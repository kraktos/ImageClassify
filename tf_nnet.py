import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_classes = 10
batch_size = 100

n_nodes_h1 = 500
n_nodes_h2 = 1500
n_nodes_h3 = 50

hm_epochs = 100

image_size = 28 * 28


def build_nn_model(data):
    h1_layer = {'weight': tf.Variable(tf.random.normal([image_size, n_nodes_h1])),
                'bias': tf.Variable(tf.random.normal([n_nodes_h1]))}

    h2_layer = {'weight': tf.Variable(tf.random.normal([n_nodes_h1, n_nodes_h2])),
                'bias': tf.Variable(tf.random.normal([n_nodes_h2]))}

    h3_layer = {'weight': tf.Variable(tf.random.normal([n_nodes_h2, n_nodes_h3])),
                'bias': tf.Variable(tf.random.normal([n_nodes_h3]))}

    out_layer = {'weight': tf.Variable(tf.random.normal([n_nodes_h3, n_classes])),
                 'bias': tf.Variable(tf.random.normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, h1_layer['weight']), h1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, h2_layer['weight']), h2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, h3_layer['weight']), h3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, out_layer['weight']), out_layer['bias'])

    return output


def train_nn(mnist):
    x = tf.compat.v1.placeholder('float', [None, image_size])
    y = tf.compat.v1.placeholder('float')

    prediction = build_nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Train the network
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

        # Evaluate the model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


def get_data():
    return input_data.read_data_sets("/tmp/data/", one_hot=True)


def main():
    print("hello")
    mnist = get_data()
    train_nn(mnist)


if __name__ == "__main__":
    main()

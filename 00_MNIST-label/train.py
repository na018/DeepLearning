import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from net.tf_net import \
    calculate_accuracy, calculate_loss, \
    create_simple_cnn_model, optimize_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a simple neural net to recognize number images from the MNIST dataset and appy the correct labeld')
    parser.add_argument('--batch_amount', default=200,
                        help='Amount of batches the net trains on')
    parser.add_argument('--batch_size', default=100,
                        help='Number of training samples inside one batch')
    parser.add_argument('--lr', default=0.01, help='Learning Rate')
    args = parser.parse_args()

    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_images, test_labels = mnist_data.test.images, mnist_data.test.labels
    input_size, n_classes = 784, 10

    # declare placeholder
    x_input = tf.placeholder(tf.float32, shape=[None, input_size])
    y_input = tf.placeholder(tf.float32, shape=[None, n_classes])
    # if test set dropout to false
    bool_dropout = tf.placeholder(tf.bool)

    # create neural net and receive logits
    logits, y_input = create_simple_cnn_model(x_input, y_input, bool_dropout)
    # calculate loss, optimize weights and calculate accuracy
    loss_operation = calculate_loss(logits, y_input)
    optimizer = optimize_weights(loss_operation)
    accuracy_operation = calculate_accuracy(logits, y_input)

    # start training
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # merge all summary for tensorboard
    merged_summary_operation = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter('/tmp/train', session.graph)
    test_summary_writer = tf.summary.FileWriter('/tmp/test')

    for batch_n in range(args.batch_amount):
        mnist_batch = mnist_data.train.next_batch(args.batch_size)
        train_images, train_labels = mnist_batch[0], mnist_batch[1]

        _, merged_summary = session.run([optimizer, merged_summary_operation],
                                        feed_dict={
            x_input: train_images,
            y_input: train_labels,
            bool_dropout: True
        })

        train_summary_writer.add_summary(merged_summary, batch_n)

        if batch_n % 10 == 0:
            merged_summary, _ = \
                session.run([merged_summary_operation, accuracy_operation],
                            feed_dict={
                    x_input: test_images,
                    y_input: test_labels,
                    bool_dropout: False
                })

            test_summary_writer.add_summary(merged_summary, batch_n)

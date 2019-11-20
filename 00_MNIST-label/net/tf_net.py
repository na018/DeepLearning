import tensorflow as tf
from .tf_base_net import conv, max_pool, fc


def create_simple_cnn_model(input_size=784, n_classes=10):
    # create placeholder
    x_input = tf.placeholder(tf.float32, shape=[None, input_size])
    y_input = tf.placeholder(tf.float32, shape=[None, n_classes])
    # if test set dropout to false
    bool_dropout = tf.placeholder(tf.bool)

    # reshape mnist dataset into images with (28,28) px
    x_input_reshaped = tf.reshape(x_input, \
                                  # -1: batch size can be any number, 1 channel
                                  [-1, 28, 28, 1], name='input_reshape')

    conv_1 = conv(x_input_reshaped, 64)
    pool_1 = max_pool(conv_1,)

    conv_2 = conv(pool_1, 128)
    pool_2 = max_pool(conv_2)

    flattened = tf.reshape(pool_2, [-1, 5*5*128], name='flattened')

    fc_1 = fc(flattened, 1024)

    dropout_1 = tf.layers.dropout(
        inputs=fc_1,
        rate=0.4,
        training=bool_dropout
    )

    # In Math, a Logit is a function that maps probabilities ([0, 1]) to R ((-inf, inf))
    logits_1 = fc(dropout_1, n_classes)

    return logits_1


def calculate_loss(logits, y_input):
    with tf.name.scope('loss'):
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_input, logits=logits
        )
        loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
        tf.summary.scalar('loss', loss_operation)

        return loss_operation


def optimize_weights(loss_operation):
    with tf.name.scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss_operation)

        return optimizer


def calculate_accuracy(logits, y_input):
    with tf.name.scope('correct_prediction'):
        predictions = tf.argmax(logits, 1)
        correct_preditions = tf.equal(predictions, tf.argmax(y_input, 1))

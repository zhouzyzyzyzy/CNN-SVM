import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CNN_softmax:
    def __init__(self,
                 num_classes=10,
                 conv1_filters=32,  # number of filters for 1st conv layer.
                 conv2_filters=64,  # number of filters for 2nd conv layer.
                 fc1_units=3072,  # number of neurons for 1st fully-connected layer.
                 kernel_len=5,
                 ):

        self.name = "CNN-softmax"
        self.num_classes = num_classes
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.fc1_units = fc1_units
        self.kernel_len = kernel_len
        self.channels = 1
        self.image_width = 0
        # create model of neural network with variables w and b
        self.random_normal = tf.initializers.RandomNormal()
        self.weights = {
            # missing Conv Layer 1, which will be created in fit()
            "wc1": None,
            'wc2': tf.Variable(self.random_normal([kernel_len, kernel_len, conv1_filters, conv2_filters])),
            # missing wd1, which will be created in fit()
            'wd1': None,
            'out': tf.Variable(self.random_normal([fc1_units, num_classes]))
        }
        self.biases = {
            'bc1': tf.Variable(tf.zeros([conv1_filters])),
            'bc2': tf.Variable(tf.zeros([conv2_filters])),
            'bd1': tf.Variable(tf.zeros([fc1_units])),
            'out': tf.Variable(tf.zeros([num_classes]))
        }

    def conv_net(self, x, pooling_merge_number=2, dropout_rate=0.2):

        def conv2d(x, W, b, strides=1):
            # Conv2D wrapper, with bias and relu activation.
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool2d(x, k=2):
            # MaxPool2D wrapper.
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        x = tf.reshape(x, [-1, self.image_width, self.image_width, self.channels])  # Input shape: [-1, 32, 32, 3].
        conv1 = conv2d(x, self.weights['wc1'], self.biases['bc1'])  # Output shape: [-1, 32, 32, 32].
        conv1 = maxpool2d(conv1, k=pooling_merge_number)  # Output shape: [-1, 16, 16, 32].
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = maxpool2d(conv2, k=pooling_merge_number)
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])  # Output shape: [-1, 8*8*64].
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])  # Output shape: [-1, 3072].
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, rate=dropout_rate)
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return tf.nn.softmax(out)

    def cross_entropy(self, y_pred, y_true):
        # Encode label to a one hot vector.
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        # Clip prediction values to avoid log(0) error.
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
        # Compute cross-entropy.
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

    # Accuracy metric.
    def accuracy(self, y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    def fit(self,
            x_train, y_train,
            image_width,
            optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
            training_steps=20,
            display_step=10,
            batch_size=100,
            pooling_merge_number=2,
            dropout_rate=0.2,
            channels=3):

        self.image_width = image_width
        self.channels = channels
        # create Conv Layer 1
        self.weights['wc1'] = tf.Variable(
            self.random_normal([self.kernel_len, self.kernel_len, channels, self.conv1_filters]))
        # create wd1 for weights
        self.weights['wd1'] = tf.Variable(self.random_normal(
            [((image_width // (pooling_merge_number ** 2)) ** 2) * self.conv2_filters, self.fc1_units]))

        def run_optimization(x, y):
            with tf.GradientTape() as g:
                pred = self.conv_net(x)
                loss = self.cross_entropy(pred, y)
            trainable_variables = list(self.weights.values()) + list(self.biases.values())
            gradients = g.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            # store the loss and pred to plot

        loss_list = []
        acc_list = []
        step_list = []

        # create train_data
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

        for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
            batch_y = tf.transpose(batch_y)[0]

            # Run the optimization to update W and b values.
            run_optimization(batch_x, batch_y)

            pred = self.conv_net(batch_x)
            loss = self.cross_entropy(pred, batch_y)
            acc = self.accuracy(pred, batch_y)
            step_list.append(step)
            loss_list.append(loss)
            acc_list.append(acc)
            if step % display_step == 0:
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        return (step_list, loss_list, acc_list)

    def evaluate(self, x_test, y_test):
        pred = self.conv_net(x_test)
        acc = self.accuracy(pred, y_test)
        acc = acc.numpy()
        print(f"The accuracy of {self.name} on the testset is {acc}")


class CNN_SVM:

    def __init__(self,
                 num_classes=10,
                 conv1_filters=32,  # number of filters for 1st conv layer.
                 conv2_filters=64,  # number of filters for 2nd conv layer.
                 fc1_units=3072,  # number of neurons for 1st fully-connected layer.
                 kernel_len=5,
                 penality_parameter = 1,
                 ):

        self.name = "CNN-SVM"
        self.num_classes = num_classes
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.fc1_units = fc1_units
        self.kernel_len = kernel_len
        self.channels = 1
        self.image_width = 0
        self.batch_size = 0
        self.penality_parameter = penality_parameter
        # create model of neural network with variables w and b
        self.random_normal = tf.initializers.RandomNormal()
        self.weights = {
            # missing Conv Layer 1, which will be created in fit()
            "wc1": None,
            'wc2': tf.Variable(self.random_normal([kernel_len, kernel_len, conv1_filters, conv2_filters])),
            # missing wd1, which will be created in fit()
            'wd1': None,
            'SVM': tf.Variable(self.random_normal([fc1_units, num_classes]))
        }
        self.biases = {
            'bc1': tf.Variable(tf.zeros([conv1_filters])),
            'bc2': tf.Variable(tf.zeros([conv2_filters])),
            'bd1': tf.Variable(tf.zeros([fc1_units])),
            # 'SVM': tf.Variable(tf.zeros([num_classes]))
        }

    def conv_net(self, x, pooling_merge_number=2, dropout_rate=0.2):

        def conv2d(x, W, b, strides=1):
            # Conv2D wrapper, with bias and relu activation.
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool2d(x, k=2):
            # MaxPool2D wrapper.
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        x = tf.reshape(x, [-1, self.image_width, self.image_width, self.channels])  # Input shape: [-1, 32, 32, 3].
        conv1 = conv2d(x, self.weights['wc1'], self.biases['bc1'])  # Output shape: [-1, 32, 32, 32].
        conv1 = maxpool2d(conv1, k=pooling_merge_number)  # Output shape: [-1, 16, 16, 32].
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = maxpool2d(conv2, k=pooling_merge_number)
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])  # Output shape: [-1, 8*8*64].
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])  # Output shape: [-1, 3072].
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.nn.dropout(fc1, rate=dropout_rate)
        svm = tf.matmul(fc2, self.weights['SVM'])
        return svm

    def soft_margin(self, y_pred, y_true):
        regularization_loss = tf.math.reduce_sum(tf.square(self.weights['SVM']))
        y_true = 2 * tf.one_hot(y_true, depth=self.num_classes) - 1
        hinge_loss = tf.math.reduce_sum(
            tf.square(
                tf.maximum(
                    tf.zeros([self.batch_size, self.num_classes]),
                    1 - y_true * y_pred)
            )
        )
        return tf.add(regularization_loss, self.penality_parameter * hinge_loss)

    # Accuracy metric.
    def accuracy(self, y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    def fit(self,
            x_train, y_train,
            image_width,
            optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
            training_steps=20,
            display_step=10,
            batch_size=100,
            pooling_merge_number=2,
            dropout_rate=0.2,
            channels=3):

        self.image_width = image_width
        self.channels = channels
        self.batch_size = batch_size

        # create Conv Layer 1
        self.weights['wc1'] = tf.Variable(
            self.random_normal([self.kernel_len, self.kernel_len, channels, self.conv1_filters]))
        # create wd1 for weights
        self.weights['wd1'] = tf.Variable(self.random_normal(
            [((image_width // (pooling_merge_number ** 2)) ** 2) * self.conv2_filters, self.fc1_units]))

        def run_optimization(x, y):
            with tf.GradientTape() as g:
                pred = self.conv_net(x)
                loss = self.soft_margin(pred, y)
            trainable_variables = list(self.weights.values()) + list(self.biases.values())
            gradients = g.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            # store the loss and pred to plot

        loss_list = []
        acc_list = []
        step_list = []

        # create train_data
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

        for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
            batch_y = tf.transpose(batch_y)[0]

            # Run the optimization to update W and b values.
            run_optimization(batch_x, batch_y)

            pred = self.conv_net(batch_x)
            loss = self.soft_margin(pred, batch_y)
            acc = self.accuracy(pred, batch_y)
            step_list.append(step)
            loss_list.append(loss)
            acc_list.append(acc)
            if step % display_step == 0:
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        return (step_list, loss_list, acc_list)

    def evaluate(self, x_test, y_test):
        pred = self.conv_net(x_test)
        acc = self.accuracy(pred, y_test)
        acc = acc.numpy()
        print(f"The accuracy of {self.name} on the testset is {acc}")

#class paint:
def draw(modelname , datasetname, step_list, acc_list, loss_list):
    plt.plot(step_list, acc_list)
    plt.title(f'model {modelname} on {datasetname}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(step_list, loss_list)
    plt.title(f'model {modelname} on {datasetname}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
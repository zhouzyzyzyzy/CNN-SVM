import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow import keras


class DNN_softmax:
    def __init__(self,
                 num_classes=10,  # the number of classes
                 num_input=70,  # the length of the input
                 n_hidden_1=512,  # the number of hidden 1
                 n_hidden_2=512,  # the number of hidden 2
                 ):

        self.name = "DNN-softmax"
        self.num_classes = num_classes

        # create model of neural network with variables w and b
        random_normal = tf.initializers.RandomNormal()
        self.weights = {
            'h1': tf.Variable(random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(random_normal([n_hidden_2, num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'out': tf.Variable(tf.zeros([num_classes]))
        }

    def DNN_net(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return tf.nn.softmax(out_layer)

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
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            training_steps=20,
            display_step=10,
            batch_size=100,
            ):

        def run_optimization(x, y):
            with tf.GradientTape() as g:
                pred = self.DNN_net(x)
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
            # Run the optimization to update W and b values.
            run_optimization(batch_x, batch_y)

            pred = self.DNN_net(batch_x)
            loss = self.cross_entropy(pred, batch_y)
            acc = self.accuracy(pred, batch_y)
            step_list.append(step)
            loss_list.append(loss)
            acc_list.append(acc)
            if step % display_step == 0:
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        return (step_list, loss_list, acc_list)

    def evaluate(self, x_test, y_test):
        pred = self.DNN_net(x_test)
        acc = self.accuracy(pred, y_test)
        acc = acc.numpy()
        print(f"The accuracy of {self.name} on the testset is {acc}")


class DNN_SVM:
    def __init__(self,
                 num_classes=10,  # the number of classes
                 num_input=70,  # the length of the input
                 n_hidden_1=512,  # the number of hidden 1
                 n_hidden_2=512,  # the number of hidden 2
                 ):

        self.name = "DNN-SVM"
        self.num_classes = num_classes

        # create model of neural network with variables w and b
        random_normal = tf.initializers.RandomNormal()
        self.weights = {
            'h1': tf.Variable(random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
            'SVM': tf.Variable(random_normal([n_hidden_2, num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'SVM': tf.Variable(tf.zeros([num_classes]))
        }

    def DNN_net(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        out_layer = tf.matmul(layer_2, self.weights['SVM']) + self.biases['SVM']
        return tf.nn.softmax(out_layer)

    def soft_margin(self, y_pred, y_true, batch_size):
        regularization_loss = tf.math.reduce_sum(tf.square(self.weights['SVM']))
        y_true = 2 * tf.one_hot(y_true, depth=self.num_classes) - 1
        hinge_loss = tf.math.reduce_sum(
            tf.square(
                tf.maximum(
                    tf.zeros([batch_size, self.num_classes]),
                    1 - y_true * y_pred)
            )
        )
        return tf.add(regularization_loss, 2 * hinge_loss)

    # Accuracy metric.
    def accuracy(self, y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    def fit(self,
            x_train, y_train,
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            training_steps=20,
            display_step=10,
            batch_size=100,
            ):

        def run_optimization(x, y, batch_size):
            with tf.GradientTape() as g:
                pred = self.DNN_net(x)
                loss = self.soft_margin(pred, y, batch_size)
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
            # Run the optimization to update W and b values.
            run_optimization(batch_x, batch_y, batch_size)

            pred = self.DNN_net(batch_x)
            loss = self.soft_margin(pred, batch_y, batch_size)
            acc = self.accuracy(pred, batch_y)
            step_list.append(step)
            loss_list.append(loss)
            acc_list.append(acc)
            if step % display_step == 0:
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        return (step_list, loss_list, acc_list)

    def evaluate(self, x_test, y_test):
        pred = self.DNN_net(x_test)
        acc = self.accuracy(pred, y_test)
        acc = acc.numpy()
        print(f"The accuracy of {self.name} on the testset is {acc}")


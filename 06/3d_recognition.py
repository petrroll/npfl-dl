#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._voxels = data["voxels"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False

class Network():
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name="voxels")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            last_layer = self.voxels
            self.global_step = tf.train.create_global_step()            

            # Create network
            last_layer = self.__create_layers_from_config_str(args.cnn, last_layer)

            # Predictions
            predictions_dense = tf.layers.dense(last_layer, self.LABELS)
            self.predictions = tf.argmax(predictions_dense, axis = 1)

            # Loss
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, predictions_dense, scope="labels_loss")

            # Training
            learning_rate = self.__create_exp_decay_learning_rate(args)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}

            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def __create_exp_decay_learning_rate(self, args):
        # Learning rate global
        learning_decay_rate = (args.learning_rate_final / args.learning_rate) ** (1/(args.epochs - 1)) if args.epochs > 1 else 1.0
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            self.global_step,
            args.batches_per_epoch,
            learning_decay_rate,
        )
        return learning_rate

    def __create_layers_from_config_str(self, config_string, last_layer):
        for layer_config in self.__parse_features(config_string):
            last_layer = self.__create_and_connect_layer(layer_config, last_layer)
        return last_layer


    def __create_and_connect_layer(self, config, last_layer):
        name, params = config
        if name   == "R": return tf.layers.dense(last_layer, units=params[0], activation=tf.nn.relu)
        elif name == "D": return tf.layers.dropout(last_layer, rate=0.5, training=self.is_training)
        elif name == "F": return tf.layers.flatten(last_layer)
        elif name == "M": return tf.layers.max_pooling3d(last_layer, pool_size=params[0], strides=params[1])
        elif name == "C": return tf.layers.conv3d(last_layer, 
            filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=tf.nn.relu)
        elif name == "CT": return tf.layers.conv3d_transpose(last_layer, 
            filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=tf.nn.relu)
        elif name == "CB":
            last_layer = tf.layers.conv3d(last_layer, 
                filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=None, use_bias=False)
            last_layer = tf.layers.batch_normalization(last_layer, training=self.is_training)
            return tf.nn.relu(last_layer)
        elif name == "CBT":
            last_layer = tf.layers.conv3d_transpose(last_layer, 
                filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=None, use_bias=False)
            last_layer = tf.layers.batch_normalization(last_layer, training=self.is_training)
            return tf.nn.relu(last_layer)
        else: raise Exception()

    @staticmethod      
    def __parse_features(features_string):  
        def __parse_config(config):
            name, *args = config.split("-")
            parsed_args = [int(x) if x.isdigit() else x for x in args]
            return (name, parsed_args)

        return [__parse_config(x) for x in features_string.split(",")]


    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries["train"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})



if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")

    parser.add_argument("--modelnet_dim", default=20, type=int, help="Dimension of ModelNet data.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_split", default=0.9, type=float, help="Ratio of examples to use as train.")

    parser.add_argument("--cnn", default="C-64-3-1-same,C-64-3-1-same,M-3-2,C-128-3-1-same,C-128-3-1-same,M-3-2,F", type=str, help="Description of the CNN architecture common.")

    parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Learning rate.")

    parser.add_argument("--acc_threshold", default=0.85, type=float, help="Minimum accuracy to predict & safe the data.")
    parser.add_argument("--batch_evaluation", default=False, type=bool, help="Is evaluation batched.") # Distors dev summaries, can help with OOM. 

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset("modelnet{}-train.npz".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset("modelnet{}-test.npz".format(args.modelnet_dim), shuffle_batches=False)

    # Construct the network
    args.batches_per_epoch = len(train.labels) // args.batch_size
    network = Network(threads=args.threads)
    network.construct(args)


    # Helper functions that train, test, and predict datasets. 
    def train_epoch(data):
        while not data.epoch_finished():
            voxels, labels = data.next_batch(args.batch_size)
            network.train(voxels, labels)

    def evaluate_on(data):
        acc = 0.0
        while not data.epoch_finished():
            evaluate_batch_size = args.batch_size if args.batch_evaluation else len(data.labels)
            voxels, labels = data.next_batch(evaluate_batch_size)
            batch_acc = network.evaluate("dev", voxels, labels)
            acc += batch_acc * len(labels) # average over all batches weighted by their length 
        acc /= len(data.labels)
        return acc


    def predict_and_save(data, save_suffix):
        with open("{}/3d_recognition_test_{}.txt".format(args.logdir, save_suffix), "w") as test_file:
            while not data.epoch_finished():
                voxels, _ = data.next_batch(args.batch_size)
                labels = network.predict(voxels)

                for label in labels:
                    print(label, file=test_file)

    # Train
    for i in range(args.epochs):
        train_epoch(train)
        dev_acc = evaluate_on(dev)
        print("{}|acc:{:.4f}".format(i, dev_acc))

        if dev_acc > args.acc_threshold:
            args.acc_threshold = dev_acc
            predict_and_save(test, str(dev_acc))




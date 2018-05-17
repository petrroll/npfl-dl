#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))
            return True
        return False


class Network:
    WIDTH, HEIGHT = 224, 224
    LABELS = 250

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed

        self.session = tf.Session(graph = graph, config=tf.ConfigProto(
            inter_op_parallelism_threads=threads, 
            intra_op_parallelism_threads=threads, 
            device_count=({'CPU' : 1, 'GPU' : 0} if args.forceCPU else {'CPU' : 1, 'GPU' : 1}), 
            ))

    def __create_cbn_layer(self, last_layer, filters, kernel_size, strides, padding):
        last_layer = tf.layers.conv2d(last_layer, filters, kernel_size, strides, padding, activation=None, use_bias=False)
        last_layer = tf.layers.batch_normalization(last_layer, training=self.is_training)
        last_layer = tf.nn.relu(last_layer)

        return last_layer

    def __create_cnn_layer(self, last_layer, filters, kernel_size, strides, padding):
        last_layer = tf.layers.conv2d(last_layer, filters, kernel_size, strides, padding, activation=tf.nn.relu)
        return last_layer

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            converted_images = tf.image.convert_image_dtype(self.images, tf.float32)
            encoded_images = 2 * (tf.tile(converted_images, [1, 1, 1, 3]) - 0.5)       

            # Encoder layers
            # WARNING: Causes errors on Nasnet checkpoint restoration -> skipped for now. 
            # with tf.variable_scope("encoder"):
            #    encoded_images = self.__create_cbn_layer(2*(converted_images - 0.5), 3, 3, 1, "same")
                       
            # Create nasnet
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(encoded_images, num_classes=None, is_training=False)
            
            # Enable retraining of certain layers (see graph for operation names / print-out via: https://stackoverflow.com/a/43703647/915609)
            retrain_layers = []
            for rtl in retrain_layers:
                rtl_layer = tf.get_default_graph().get_operation_by_name(rtl)
                rtl_layer.trainable = True
            
            self.nasnet_saver = tf.train.Saver()

            # Retrieve the tensor to use as input for decoder part of the network -> last conv network (see above on how to get it's name)
            nasnet_output = tf.get_default_graph().get_tensor_by_name("final_layer/Relu:0")
            
            # Decoder layers
            with tf.variable_scope("decoder"):             
                # Adding own c[nb]n layer slowed the network's progress and/or increased overfitting
                # decoded_results = self.__create_cnn_layer(nasnet_output, filters=32, kernel_size=3, strides=2, padding="same")
                # decoded_results = self.__create_cnn_layer(decoded_results, filters=32, kernel_size=3, strides=2, padding="same")

                # Using nasnet's convolution output actually performs worse than using features for our network (not enough data)
                # flatten_decoded_result = tf.layers.flatten(decoded_results)

                # Prediction and output
                # Adding own dense layer actually increased overfitting on our data after certain accuracy was achieved
                # output_layer = tf.layers.dense(flatten_decoded_result, self.LABELS, activation=None, name="output_layer")
                output_layer = features
            
            # Predictions
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")

            with tf.control_dependencies(update_ops):
                # Uncomment if decoder is non-empty
                # self.training_enc_dec = tf.train.AdamOptimizer().minimize(self.loss, 
                    # var_list=trainable_vars , global_step=global_step, name="trainingDecoderEncoder")
                self.training_full = tf.train.AdamOptimizer().minimize(self.loss, 
                    global_step=global_step, name="trainingFull")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

    def train_batch(self, images, labels, train_full = False):
        self.session.run([self.training_full if train_full else self.training_enc_dec, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {self.images: batch_images, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)


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
    parser.add_argument("--nasnet", default="nets/nasnet/model.ckpt", type=str, help="NASNet checkpoint path.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--forceCPU", default=False, type=bool, help="Force coputation graph on CPU.")
    parser.add_argument("--accLimitTrainWhole", default=0.6, type=float, help="Threshold of last epoch's dev accuracy to start training the whole network.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("nsketch-train.npz")
    dev = Dataset("nsketch-dev.npz", shuffle_batches=False)
    test = Dataset("nsketch-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        last_acc = 0.0
        while not train.epoch_finished():
            images, labels = train.next_batch(args.batch_size)
            network.train_batch(images, labels, last_acc >= args.accLimitTrainWhole)

        last_acc = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(last_acc))

    # Predict test data
    with open("{}/nsketch_transfer_test.txt".format(args.logdir), "w") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)

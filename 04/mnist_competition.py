#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, batches_per_epoch):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder_with_default(False, [], name="is_training")
            
            # Computation
            last_layer = self.images
            global_step = tf.train.create_global_step()


            # Layers are separated by a comma and can be:
            # - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
            #   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
            # - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
            # - F: Flatten inputs
            # - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
            for (name, params) in self.__parse_features(args.cnn):
                if name   == "R": last_layer = tf.layers.dense(last_layer, units=params[0], activation=tf.nn.relu)
                elif name == "D": last_layer = tf.layers.dropout(last_layer, rate=0.5, training=self.is_training)
                elif name == "F": last_layer = tf.layers.flatten(last_layer)
                elif name == "M": last_layer = tf.layers.max_pooling2d(last_layer, pool_size=params[0], strides=params[1])
                elif name == "C": last_layer = tf.layers.conv2d(last_layer, filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=tf.nn.relu)
                elif name == "CB":
                    last_layer = tf.layers.conv2d(last_layer, filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=None, use_bias=False)
                    last_layer = tf.layers.batch_normalization(last_layer, training=self.is_training)
                    last_layer = tf.nn.relu(last_layer)
                else: raise Exception()


            output_layer = tf.layers.dense(last_layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)
            
            # Training
            learning_decay_rate = (args.learning_rate_final / args.learning_rate) ** (1/(args.epochs - 1))
            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                global_step,
                batches_per_epoch,
                learning_decay_rate,
            )

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
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

    @staticmethod      
    def __parse_features(features_string):  
        def __parse_config(config):
            name, *args = config.split("-")
            parsed_args = [int(x) if x.isdigit() else x for x in args]
            return (name, parsed_args)

        return [__parse_config(x) for x in features_string.split(",")]

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels})
        return accuracy

    def infer(self, images):
        return self.session.run(self.predictions, {self.images: images})
        


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logname", default="mnist_comp", type=str, help="Logname prefix.")
    parser.add_argument("--printout", default=False, type=bool, help="Infer and printout test data.")


    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    
    parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Learning rate.")

    parser.add_argument("--cnn", default="C-64-3-1-same,M-3-2,C-16-3-1-same,F,R-500,D", type=str, help="Description of the CNN architecture.")


    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}-{}".format(
        args.logname,
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
    source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    batches_per_epoch = mnist.train.num_examples // args.batch_size
    network = Network(threads=args.threads)
    network.construct(args, batches_per_epoch)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        dev_acc = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        print(f"{i}:{dev_acc*100:.4f}")

    if args.printout:
        test_labels = network.infer(mnist.test.images)
        with open("mnist_competition_test.txt", "w") as test_file: 
            for label in test_labels: 
                print(label, file=test_file) 

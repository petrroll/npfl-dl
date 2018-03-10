#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            last_layer = self.observations


            for i in range(args.layers):
                last_layer = tf.layers.dense(
                    last_layer, args.hidden_layer, 
                    activation={
                        "relu": tf.nn.relu,
                        "tanh" : tf.nn.tanh,
                        "sigmoid" : tf.nn.sigmoid,
                        "none" : None,
                        }[args.activation], 
                    name=("hidden_layer" + str(i))
                )

            output_layer = tf.layers.dense(last_layer, self.ACTIONS, name="output_layer")

            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            # Global step
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.summaries = [tf.contrib.summary.scalar("train/loss", loss),
                                  tf.contrib.summary.scalar("train/accuracy", accuracy)]

            self.accuracy = accuracy

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        return self.session.run([self.training, self.summaries, self.accuracy], {self.observations: observations, self.labels: labels})[2]

    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default=6, type=int, help="Number of batches.")
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--hidden_layer", default=5, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
    parser.add_argument("--activation", default="tanh", type=str, help="Activation function.")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[0:4]])
            labels.append(int(columns[4]))
    observations, labels = np.array(observations), np.array(labels)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    data_size = len(observations)
    batch_size = int(data_size // args.batches)
    # Train
    for i in range(args.epochs):
        perm = np.random.permutation(data_size)
        
        rand_obs = observations[perm]
        rand_labels = labels[perm]

        for j in range(data_size // batch_size):
            acc = network.train(
                rand_obs[j*batch_size:(j+1)*batch_size], 
                rand_labels[j*batch_size:(j+1)*batch_size]
                )

    # Save the network
    network.save("gym_cartpole/model")

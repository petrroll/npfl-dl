#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_tags):
        with self.session.graph.as_default():
            if args.recodex:
                tf.get_variable_scope().set_initializer(tf.glorot_uniform_initializer(seed=42))

            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")

            # TODO(we): Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell).
            if args.rnn_cell == "RNN":
                rnn_creation = tf.nn.rnn_cell.BasicRNNCell
            elif args.rnn_cell == "LSTM":
                rnn_creation = tf.nn.rnn_cell.BasicLSTMCell
            elif args.rnn_cell == "GRU":
                rnn_creation = tf.nn.rnn_cell.GRUCell
            else: rnn_creation = None

            rnn_fwd = rnn_creation(args.rnn_cell_dim)
            rnn_bck = rnn_creation(args.rnn_cell_dim)

            # TODO(we): Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            embeddings = tf.get_variable("embdgs", [num_words, args.we_dim])

            # TODO(we): Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            embeded_words = tf.nn.embedding_lookup(embeddings, self.word_ids)

            # TODO(we): Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            bin_rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fwd, rnn_bck, embeded_words, sequence_length = self.sentence_lens, dtype=tf.float32)

            # TODO(we): Concatenate the outputs for fwd and bwd directions (in the third dimension).
            memories_output = tf.concat(bin_rnn_outputs, 2)

            # TODO(we): Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            output = tf.layers.dense(memories_output, num_tags, activation=None)

            # TODO(we): Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Loss and predictions

            # TODO: Compute log likelihood and transition parameters using tf.contrib.crf.crf_log_likelihood
            # and store the mean of sentence losses into `loss`.
            # learns transition probabilities through training (from golden data)
            # ... uses these probabilities to calculate crf probability of current sequence
            log_likelyhoods, transitions = tf.contrib.crf.crf_log_likelihood(output, self.tags, sequence_lengths=self.sentence_lens)
            loss = tf.reduce_mean(-log_likelyhoods)

            # TODO: Compute the CRF predictions into `self.predictions` with `crf_decode`.
            self.predictions, _ = tf.contrib.crf.crf_decode(output, transitions, sequence_length=self.sentence_lens)

            # Training
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            sentence_lens, word_ids = train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids = dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids[dataset.FORMS],
                              self.tags: word_ids[dataset.TAGS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-cac-train.txt", max_sentences=5000)
    dev = morpho_dataset.MorphoDataset("czech-cac-dev.txt", train=train, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.TAGS].words))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))

#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import timit_mfcc26_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_phones, mfcc_dim):
        with self.session.graph.as_default():
            # Inputs
            self.mfcc_lens = tf.placeholder(tf.int32, [None])
            self.mfccs = tf.placeholder(tf.float32, [None, None, mfcc_dim])
            self.phone_lens = tf.placeholder(tf.int32, [None])
            self.phones = tf.placeholder(tf.int32, [None, None])

            # Create sparse repre for phonemes
            idx = tf.where(tf.not_equal(self.phones, 0))
            sparse_phones = tf.SparseTensor(idx, tf.gather_nd(self.phones, idx), tf.cast(tf.shape(self.phones), tf.int64)) # Can't explain casts to int64

            # Prepend one bidirRNN.
            if args.two_layers:
                with tf.variable_scope("optional_rnn"):
                    bin_rnn_outputs_opt, _ = tf.nn.bidirectional_dynamic_rnn(
                     tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim), tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim), 
                     self.mfccs, sequence_length = self.mfcc_lens, 
                     dtype=tf.float32)

                rnn_input = tf.concat(bin_rnn_outputs_opt, axis=2) # doubling the dimensionality 
            else:
                rnn_input = self.mfccs


            # Bidir RNN to process mfccs and create phoneme sequences
            bin_rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim), tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim), 
                rnn_input, sequence_length = self.mfcc_lens, 
                dtype=tf.float32)

            # Concat fwd and bck, create output via linear layer without activation
            rnn_outputs = tf.concat(bin_rnn_outputs, axis=2)
            output = tf.layers.dense(rnn_outputs, num_phones + 1, activation=None)
            output = tf.transpose(output, [1, 0, 2]) # loss and decoder expect [max_time, batch_size, ...] not [b_s, m_t, ...]

            # TODO: Computation and training. The rest of the template assumes
            # the following variables:
            # - `losses`: vector of losses, with an element for each example in the batch
            # - `edit_distances`: vector of edit distances, with an element for each batch example

            # Produce losses and predictions 
            losses = tf.nn.ctc_loss(sparse_phones, output, self.mfcc_lens)
            predictions, _ = tf.nn.ctc_beam_search_decoder(output, self.mfcc_lens) if args.beam_decoding else tf.nn.ctc_greedy_decoder(output, self.mfcc_lens)
            predictions = predictions[0] # It's a single-element list

            # Better prediction for inference
            beam_predictions, _ = tf.nn.ctc_beam_search_decoder(output, self.mfcc_lens)
            beam_predictions = beam_predictions[0] 

            # Create dense predictions, used for inference
            self.predictions_dense = tf.sparse_to_dense(beam_predictions.indices, beam_predictions.dense_shape, beam_predictions.values)

            # Produce edit distance betweeen predictions and golden data
            edit_distances = tf.edit_distance(predictions, tf.cast(sparse_phones, tf.int64))

            # Training
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(tf.reduce_mean(losses), global_step=global_step, name="training")

            # Summaries
            self.current_edit_distance, self.update_edit_distance = tf.metrics.mean(edit_distances)
            self.current_loss, self.update_loss = tf.metrics.mean(losses)
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/edit_distance", self.update_edit_distance)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/edit_distance", self.current_edit_distance)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            mfcc_lens, mfccs, phone_lens, phones = train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.mfcc_lens: mfcc_lens, self.mfccs: mfccs,
                              self.phone_lens: phone_lens, self.phones: phones})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            mfcc_lens, mfccs, phone_lens, phones = dataset.next_batch(batch_size)
            self.session.run([self.update_edit_distance, self.update_loss],
                             {self.mfcc_lens: mfcc_lens, self.mfccs: mfccs,
                              self.phone_lens: phone_lens, self.phones: phones})
        return self.session.run([self.current_edit_distance, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        phonemes = []

        while not dataset.epoch_finished():
            mfcc_lens, mfccs, _, _ = dataset.next_batch(batch_size)
            phonemes_batch = self.session.run(self.predictions_dense,
                             {self.mfcc_lens: mfcc_lens, self.mfccs: mfccs})
            
            phonemes.extend(phonemes_batch)
        return phonemes

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--rnn_cell_dim", default=256, type=int, help="RNN cell dimension.")
    parser.add_argument("--beam_decoding", default=False, type=bool, help="Use beam decoding instead of greedy (slower, better) for train & dev.")
    parser.add_argument("--two_layers", default=True, type=bool, help="Use two bidirRNN layers.")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    timit = timit_mfcc26_dataset.TIMIT("timit-mfcc26.pickle")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(timit.phones), timit.mfcc_dim)


    # Predict test data
    def predict(acc):
        phonemes = network.predict(timit.test, args.batch_size)
        with open("{}/speech_recognition_test_{:.4f}.txt".format(args.logdir, acc), "w") as test_file:        
            for sentence in phonemes:
                # Translate phonemes indexes to actual phonemes
                translated_phonemes = [timit.phones[x] for x in sentence if not x == 0]
                print(" ".join(translated_phonemes), file=test_file)

    # Train
    best_acc = 0.6
    for i in range(args.epochs):
        network.train_epoch(timit.train, args.batch_size)
        edit_distance = network.evaluate("dev", timit.dev, args.batch_size)

        print("{}|{:.2f}".format(i, edit_distance))
        if edit_distance < best_acc:
            best_acc = edit_distance
            predict(edit_distance)




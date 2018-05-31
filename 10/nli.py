#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import nli_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))


    def with_bn(self, last_layer, activation, is_training):
        last_layer = tf.layers.batch_normalization(last_layer, training=is_training)
        last_layer = activation(last_layer)

        return last_layer

    def construct(self, args, num_words, num_chars, num_languages, num_tags, num_prompts):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            
            self.languages = tf.placeholder(tf.int32, [None], name="languages")

            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.levels = tf.placeholder(tf.int32, [None], name="levels")
            self.prompts = tf.placeholder(tf.int32, [None], name="prompts")

            self.is_training = tf.placeholder_with_default(False, [], name="is_training")

            one_hot_tags = tf.one_hot(self.tags, num_tags)
            one_hot_prompts = tf.one_hot(self.prompts, num_prompts)

            # Word embeddings
            with tf.variable_scope("word_embeds"):
                word_embeddings = tf.get_variable("wrd_embdgs", [num_words, args.we_dim])
                embeded_words = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            # Char embeddings
            with tf.variable_scope("char_embeds"):
                char_embeddings = tf.get_variable("char_embdgs", [num_chars, args.cle_dim])
                embeded_chars = tf.nn.embedding_lookup(char_embeddings, self.charseqs)

                _, (chars_fwd_state, chars_bck_state) = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim), 
                tf.nn.rnn_cell.GRUCell(args.cle_dim), 
                embeded_chars, dtype=tf.float32, sequence_length=self.charseq_lens)
                words_cle = tf.add(chars_fwd_state, chars_bck_state)

                cle = tf.nn.embedding_lookup(words_cle, self.charseq_ids)
            
            # Embeddings
            with tf.variable_scope("embeds"):
                embedings = tf.concat([embeded_words, cle, one_hot_tags], axis=2)

            # Essay processing
            with tf.variable_scope("process"):
                # RNN
                bin_rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.rnn_dim), 
                tf.nn.rnn_cell.GRUCell(args.rnn_dim), 
                embedings, dtype=tf.float32, sequence_length=self.sentence_lens)

                bin_rnn_outputs = tf.concat(bin_rnn_outputs, 2)
                bin_rnn_outputs = tf.layers.dropout(bin_rnn_outputs, rate=0.5, training=self.is_training)

                # 1-d convolution
                conv1 = tf.layers.conv1d(bin_rnn_outputs, args.conv_filters, 7, strides=1, padding="same", use_bias=False)
                conv1 = self.with_bn(conv1, tf.nn.relu, self.is_training)

                max2 = tf.layers.max_pooling1d(conv1, pool_size=5, strides=5)
                
                conv3 = tf.layers.conv1d(max2, 2*args.conv_filters, 7, strides=1, padding="same", use_bias=False)
                conv3 = self.with_bn(conv3, tf.nn.relu, self.is_training)

                max4 = tf.layers.max_pooling1d(conv3, pool_size=5, strides=5, )

                # Reduces current [batch, sentence_length, repre_dim] -> [batch, repre_dim]
                sent_lengths = tf.tile(tf.expand_dims(tf.cast(self.sentence_lens, tf.float32), -1), [1, 2*args.conv_filters])
                sent_summed_states = tf.reduce_sum(max4, axis=1)
                essay_repre = tf.divide(sent_summed_states, sent_lengths)


            essay_info = tf.concat([
                essay_repre, 
                tf.expand_dims(tf.cast(self.sentence_lens, tf.float32), -1), 
                tf.expand_dims(tf.cast(self.levels, tf.float32), -1), 
                tf.cast(one_hot_prompts, tf.float32)
                ], axis=1)
            essay_info_processed = tf.layers.dense(essay_info, num_languages, activation=tf.nn.relu)

            # Prediction and logits

            logits = tf.layers.dense(essay_info_processed, num_languages, activation=None)
            self.predictions = tf.argmax(logits, axis=1)

            # Training
            global_step = tf.train.create_global_step()
            loss = tf.losses.sparse_softmax_cross_entropy(self.languages, logits, scope="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.languages, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.size(self.sentence_lens))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
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
            print("Doing stuff")
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages,
                              self.tags: tags, self.levels: levels, self.prompts: prompts,
                              self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages,
                              self.tags: tags, self.levels: levels, self.prompts: prompts})

        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        languages = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, _ = \
                dataset.next_batch(batch_size)
            languages.extend(self.session.run(self.predictions,
                                              {self.sentence_lens: sentence_lens,
                                               self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                                               self.word_ids: word_ids, self.charseq_ids: charseq_ids, 
                                               self.tags: tags, self.levels: levels, self.prompts: prompts}))

        return languages


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    parser.add_argument("--cle_dim", default=128, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--conv_filters", default=16, type=int, help="Convolution filters.")


    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = nli_dataset.NLIDataset("nli-train.txt")
    dev = nli_dataset.NLIDataset("nli-dev.txt", train=train, shuffle_batches=False)
    test = nli_dataset.NLIDataset("nli-test.txt", train=train, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.vocabulary("words")), len(train.vocabulary("chars")), len(train.vocabulary("languages")), len(train.vocabulary("tags")), len(train.vocabulary("prompts")))

    # Predict test data
    def predict_data(acc):
        with open("{}/nli_test_{:.4f}.txt".format(args.logdir, acc), "w", encoding="utf-8") as test_file:
            languages = network.predict(test, args.batch_size)
            for language in languages:
                print(test.vocabulary("languages")[language], file=test_file)


    # Train
    best_acc = 0.5
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        acc = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(acc))

        if acc > best_acc:
            predict_data(acc)
            best_acc = acc




#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        # feed the 2D array with slices corresponding to individial window displacements 
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i] 
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False

    def get_capitilized_text(self, capitalization):
        capitilized_text = ""
        for i in range(len(capitalization)):
            curr_letter = self.text[i]
            capitilized_text += curr_letter.upper() if capitalization[i] else curr_letter
        return capitilized_text

    def save_capitilezed_text(self, capitalization, filename = "uppercase_result_test.txt"):
        # Save capitilized data
        capitilized_text = self.get_capitilized_text(capitalization)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(capitilized_text)


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, batches_per_epoch):
        with self.session.graph.as_default():
            window_size = 2 * args.window + 1

            # Inputs
            self.windows = tf.placeholder(tf.int32, [None, window_size], name="windows")
            self.labels = tf.placeholder(tf.int32, [None], name="labels") # Or you can use tf.int32

            input_layer = tf.one_hot(self.windows, depth = args.alphabet_size, axis = 1)
            input_layer = tf.layers.flatten(input_layer)

            last_layer = tf.to_float(input_layer)
            activation_fun={
                "relu": tf.nn.relu,
                "tanh" : tf.nn.tanh,
                "sigmoid" : tf.nn.sigmoid,
                "none" : None,
                }[args.activation] 

            for i in range(args.layers):
                last_layer = tf.layers.dense(last_layer, args.alphabet_size * window_size, name = f"hidden_{i}", activation=activation_fun)
                last_layer = tf.layers.dropout(last_layer, rate = args.dropout)

            # output layers
            # When the output layer has an activation function (e.g. ReLU) the network is unable to learn anything probably
            # ..due to it squashing all information that could be used to compute a gradient. 
            output_layer = tf.layers.dense(last_layer, 2, name="output_layer", activation=None) 
            self.predictions = tf.argmax(output_layer, axis=1, name="actions", output_type=tf.int32)

            # training
            global_step = tf.train.create_global_step()
            learning_decay_rate = (args.learning_rate_final / args.learning_rate) ** (1/(args.epochs - 1))
            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                global_step,
                batches_per_epoch,
                learning_decay_rate,
            )

            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=output_layer, scope="loss")
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, name="training")


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

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]], {self.windows: windows, self.labels: labels})

    def evaluate(self, dataset, windows, labels):
        return self.session.run({"sum" : self.summaries[dataset], "acc": self.accuracy}, {self.windows: windows, self.labels: labels})["acc"]

    def infer(self, windows):
        return self.session.run(self.predictions, {self.windows: windows})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=49, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=1024 , type=int, help="Batch size.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=6, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=5, type=int, help="Size of the window to use.")
    parser.add_argument("--layers", default=3, type=int, help="Number of hidden layers.")
    parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate on hidden layers.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Learning rate.")
    parser.add_argument("--activation", default="relu", type=str, help="Activation function.")


    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train.alphabet)

    # Construct the network
    batches_per_epoch = len(train.text) // args.batch_size
    network = Network(threads=args.threads)
    network.construct(args, batches_per_epoch)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            network.train(windows, labels)

        dev_windows, dev_labels = dev.all_data()
        dev_acc = network.evaluate("dev", dev_windows, dev_labels)
        print(f"Acc: {dev_acc}")

    test_windows, _ = test.all_data()
    capitalization = network.infer(test_windows)
    test.save_capitilezed_text(capitalization)




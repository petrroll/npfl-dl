#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._shuffle = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle else range(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm], self._masks[batch_perm]


    def next_batch_images(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle else range(len(self._images))
            return True
        return False

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
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            last_layer = self.images
            global_step = tf.train.create_global_step()            

            # Create common & task specific parts of the network
            for layer_config in self.__parse_features(args.cnn_c):
                last_layer = self.create_and_connect_layer(layer_config, last_layer)

            last_mask_layer = last_layer
            last_label_layer = last_layer

            for layer_config in self.__parse_features(args.cnn_l):
                last_label_layer = self.create_and_connect_layer(layer_config, last_label_layer)

            for layer_config in self.__parse_features(args.cnn_m):
                last_mask_layer = self.create_and_connect_layer(layer_config, last_mask_layer)

            # Masks predictions
            masks_predictions_dense = tf.layers.dense(last_mask_layer, self.HEIGHT * self.WIDTH * 2)
            masks_predictions_reshaped =  tf.reshape(masks_predictions_dense, [-1, self.HEIGHT, self.WIDTH, 2])
            self.masks_predictions = tf.argmax(masks_predictions_reshaped, axis = 3)
            # Follwing is required because summary code bellow expects shape: [None, 28, 28, 1] (not [None, 28, 28]) and type float32
            self.masks_predictions =  tf.reshape(tf.cast(self.masks_predictions, tf.float32), [-1, self.HEIGHT, self.WIDTH, 1])

            # Labels predictions
            labels_predictions_dense = tf.layers.dense(last_label_layer, self.LABELS)
            self.labels_predictions = tf.argmax(labels_predictions_dense, axis = 1)

            # Learning rate global
            learning_decay_rate = (args.learning_rate_final / args.learning_rate) ** (1/(args.epochs - 1))
            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                global_step,
                batches_per_epoch,
                learning_decay_rate,
            )

            # Mask and labels loss
            labels_loss = tf.losses.sparse_softmax_cross_entropy(self.labels, labels_predictions_dense, scope="labels_loss")
            mask_loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.masks, tf.int64), masks_predictions_reshaped, scope="mask_loss")

            # Unified loss and training
            loss = tf.add(labels_loss, mask_loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")
            
            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions), 
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            self.iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/loss_labels", labels_loss),
                                           tf.contrib.summary.scalar("train/loss_masks", mask_loss),
                                           tf.contrib.summary.scalar("train/accuracy_labels", self.accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/loss_labels", labels_loss),
                                               tf.contrib.summary.scalar(dataset+"/loss_masks", mask_loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy_labels", self.accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", self.iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    # Layers are separated by a comma and can be:
    # - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
    #   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
    # - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
    # - F: Flatten inputs
    # - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
    # - CB-...: the same as convolutional layer, only with batch normalization.
    def create_and_connect_layer(self, config, last_layer):
        name, params = config
        if name   == "R": return tf.layers.dense(last_layer, units=params[0], activation=tf.nn.relu)
        elif name == "D": return tf.layers.dropout(last_layer, rate=0.5, training=self.is_training)
        elif name == "F": return tf.layers.flatten(last_layer)
        elif name == "M": return tf.layers.max_pooling2d(last_layer, pool_size=params[0], strides=params[1])
        elif name == "C": return tf.layers.conv2d(last_layer, filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=tf.nn.relu)
        elif name == "CB":
            last_layer = tf.layers.conv2d(last_layer, filters=params[0], kernel_size=params[1], strides=params[2], padding=params[3], activation=None, use_bias=False)
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

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        _, acc, iou = self.session.run((self.summaries[dataset], self.accuracy, self.iou),
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})
        return (acc, iou)

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logname", default="1", type=str, help="Logname prefix.")
    parser.add_argument("--printout", default=True, type=bool, help="Infer and printout test data.")

    parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Learning rate.")

    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--predict_batch_size", default=512, type=int, help="Batch size for prediction.")

    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--cnn_c", default="C-64-3-1-same,C-64-3-1-same,M-3-2,C-128-3-1-same,C-128-3-1-same,M-3-2", type=str, help="Description of the CNN architecture common.")
    parser.add_argument("--cnn_m", default="C-128-3-1-same,C-128-3-1-same,F,R-1024,D", type=str, help="Description of the CNN architecture mask part.")
    parser.add_argument("--cnn_l", default="F,R-500,D", type=str, help="Description of the CNN architecture label part.")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        args.logname
        # Started having too long parameters -> can't use them to create to logname 
        #",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
        #          for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz", False)

    # Construct the network
    batches_per_epoch = len(train.labels) // args.batch_size
    network = Network(threads=args.threads)
    network.construct(args, batches_per_epoch)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)

        acc, iou = network.evaluate("dev", dev.images, dev.labels, dev.masks)
        print(f"{i}|acc:{acc:.4f}|iou:{iou:.4f}")

    if args.printout:
        with open(f"{args.logname}_fashion_masks_test.txt", "w") as test_file:
            while not test.epoch_finished():
                images = test.next_batch_images(args.predict_batch_size)
                labels, masks = network.predict(images)
                for i in range(len(labels)):
                    print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)

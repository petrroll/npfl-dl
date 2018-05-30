#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    HEIGHT, WIDTH = 28, 28

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

    def construct(self, args):
        self.z_dim = args.z_dim

        with self.session.graph.as_default():
            if args.recodex:
                tf.get_variable_scope().set_initializer(tf.glorot_uniform_initializer(seed=42))

            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1])
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])

            # Generator
            # Takes a latent vector and generates an output image using the trained 
            # ..input images distribution. 
            def generator(z):
                # TODO: Define a generator as a sequence of:
                # - batch normalized dense layer with 1024 neurons and ReLU activation
                # - batch normalized dense layer with 7 * 7 * 64 neurons and ReLU activation
                # - change shape to a batch of images with size 7 x 7 and 64 channels
                # - batch normalized conv2d_transpose with 32 output channels, kernel size 5,
                #   stride 2, "same" padding and ReLU activation
                # - (non-normalized) conv2d_transpose with 1 output channel, kernel size 5,
                #   stride 2, "same" padding and sigmoid activation
                # Return the result.
                #
                # Note that batch normalization should be used on inputs
                # without bias (`use_bias=False`) and that activation should be
                # applied after the batch normalization. Also use `training=True`
                # all the time (i.e., never use the saved estimates of moments)
                # in the batch normalization.
                dense_1 = tf.layers.dense(z, 1024, activation=None, use_bias=False)
                dense_1 = self.with_bn(dense_1, tf.nn.relu, is_training=True)

                dense_2 = tf.layers.dense(dense_1, 7*7*64, activation=None, use_bias=False)
                dense_2 = self.with_bn(dense_2, tf.nn.relu, is_training=True)

                image_3 = tf.reshape(dense_2, [-1, 7, 7, 64])   

                upscale_4 = tf.layers.conv2d_transpose(image_3, filters=32, kernel_size=5, strides=2, padding="same", activation=None, use_bias=False)
                upscale_4 = self.with_bn(upscale_4, tf.nn.relu, is_training=True)

                upscale_5 = tf.layers.conv2d_transpose(upscale_4, filters=1, kernel_size=5, strides=2, padding="same", activation=tf.nn.sigmoid)

                return upscale_5

            with tf.variable_scope("generator"):
                # TODO(GAN): Define `self.generated_images` as a result of `generator` applied to `self.z`.
                self.generated_images = generator(self.z)

            # Discriminator
            # Tries to predict whether an input image was generated by a generator or
            # ..if it was drawn directly from image examples. 
            # The discriminator uses batch-norm not only to normalize but to see information
            # ..about the current batch distribution. If it's different than usually (i.e. 
            # ..if the generator didn't generate the batch from full distro but had a mode collapse
            # ..(e.g. generated only 3s and no 6s in mist)) the batch normalization will produce different
            # ..distribution of activations (it will still squeeze the distrib. using current batch to N(0,1) 
            # ..but will then transform it using (even on the true images learned) std_dev and bias -> different activations)
            def discriminator(image):
                # TODO: Define a discriminator as a sequence of:
                # - batch normalized conv2d with 32 output channels, kernel size 5,
                #   "same" padding and ReLU activation
                # - max pooling layer with kernel size 2 and stride 2
                # - batch normalized conv2d with 64 output channels, kernel size 5,
                #   "same" padding and ReLU activation
                # - max pooling layer with kernel size 2 and stride 2
                # - flattening layer
                # - batch normalized dense layer with 1024 neurons and ReLU activation
                # - (non-normalized) dense layer with 1 neuron without activation.
                #
                # Consider the last hidden layer output to be the logit of whether the input
                # images comes from real data. Change its shape to remove the last dimension
                # (i.e., [batch_size] instead of [batch_size, 1]) and return it.
                #
                # Same considerations as in `generator` regarding the batch normalization apply.

                conv_1 = tf.layers.conv2d(image, filters=32, kernel_size=5, padding="same", activation=None, use_bias=False)
                conv_1 = self.with_bn(conv_1, tf.nn.relu, is_training=True)

                max_p_2 = tf.layers.max_pooling2d(conv_1, 2, 2, padding="same")
                
                conv_3 = tf.layers.conv2d(max_p_2, filters=64, kernel_size=5, padding="same", activation=None, use_bias=False)
                conv_3 = self.with_bn(conv_3, tf.nn.relu, is_training=True)
                
                max_p_4 = tf.layers.max_pooling2d(conv_3, 2, 2, padding="same")
                flat_5 = tf.layers.flatten(max_p_4)
                
                dense_6 = tf.layers.dense(flat_5, 1024, activation=None, use_bias=False)
                dense_6 = self.with_bn(dense_6, tf.nn.relu, is_training=True)
                
                logits = tf.layers.dense(dense_6, 1, activation=None)

                return tf.squeeze(logits)

            # Call discriminator for true images and generated images using the same network 
            # ..-> the same variable scope with reuse = True.
            with tf.variable_scope("discriminator"):
                # TODO(GAN): Define `discriminator_logit_real` as a result of
                # `discriminator` applied to `self.images`.
                discriminator_logit_real = discriminator(self.images)


            with tf.variable_scope("discriminator", reuse = True):
                # TODO(GAN): Define `discriminator_logit_fake` as a result of
                # `discriminator` applied to `self.generated_images`.
                #
                # TODO(GAN): Note the discriminator is called in the same variable
                # scope as several lines above -- it will try to utilize the
                # same variables. In order to allow reusing them, we need to explicitly
                # pass the `reuse=True` flag.
                discriminator_logit_fake = discriminator(self.generated_images)

            # Losses
            # The loss for discriminator is a sigmoid cross-entropy on predicted non-zero 
            # ..fake inputs and predicted non-ones on the true inputs.
            # For generator it's predicted non-ones by the discriminator on fake inputs. It
            # ..doesn't care about the performance of discriminator on the true inputs.

            # TODO(GAN): Define `self.discriminator_loss` as a sum of
            # - sigmoid cross entropy loss with gold labels of ones (1.0) and discriminator_logit_real
            # - sigmoid cross entropy loss with gold labels of zeros (0.0) and discriminator_logit_fake
            loss_for_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_logit_real), discriminator_logit_real)
            loss_for_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_logit_fake), discriminator_logit_fake)
            self.discriminator_loss = tf.add(loss_for_real, loss_for_fake)

            # TODO(GAN): Define `self.generator_loss` as a sigmoid cross entropy
            # loss with gold labels of ones (1.0) and discriminator_logit_fake.
            self.generator_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_logit_fake), discriminator_logit_fake)

            # Training
            # Train discriminator and generator using separate optimizers. Always update only their
            # ..variables and use their respective loss.
            global_step = tf.train.create_global_step()

            # TODO(GAN): Create `self.discriminator_training` as an AdamOptimizer.minimize
            # for discriminator_loss and variables in "discriminator" namespace.
            # Do *not* (otherwise it would increment two times during one network.train call 
            # due to there being two optimizers) pass global_step as argument to AdamOptimizer.minimize.
            self.discriminator_training = tf.train.AdamOptimizer().minimize(self.discriminator_loss, var_list=tf.global_variables("discriminator"))

            # TODO(GAN): Create `self.generator_training` as an AdamOptimizer.minimize
            # for generator_loss and variables in "generator" namespace.
            # This time *do* pass global_step as argument to AdamOptimizer.minimize.
            self.generator_training = tf.train.AdamOptimizer().minimize(self.generator_loss, global_step=global_step, var_list=tf.global_variables("generator"))

            # Summaries
            discriminator_accuracy = tf.reduce_mean(tf.to_float(tf.concat([
                tf.greater(discriminator_logit_real, 0), tf.less(discriminator_logit_fake, 0)], axis=0)))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.discriminator_summary = [tf.contrib.summary.scalar("gan/discriminator_loss", self.discriminator_loss),
                                              tf.contrib.summary.scalar("gan/discriminator_accuracy", discriminator_accuracy)]
                self.generator_summary = tf.contrib.summary.scalar("gan/generator_loss", self.generator_loss)

            self.generated_image_data = tf.placeholder(tf.float32, [None, None, 1])
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.generated_image_summary = tf.contrib.summary.image("gan/generated_image",
                                                                        tf.expand_dims(self.generated_image_data, axis=0))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def sample_z(self, batch_size):
        # TODO(GAN): Return uniform random noise in -1, 1 range using `np.random.uniform`
        # call, with shape [batch_size, self.z_dim].
        return np.random.uniform(-1, 1, [batch_size, self.z_dim])

    def train(self, images):
        # TODO(GAN): In first self.session.run, evaluate self.discriminator_training,
        # self.discriminator_summary and self.discriminator_loss using
        # `images` as `self.images` and noise sampled with `self.sample_z` as `self.z`.
        disc_loss = self.session.run(
            [self.discriminator_training, self.discriminator_summary, self.discriminator_loss],
            {self.images: images, self.z: self.sample_z(images.shape[0])}
        )[-1]

        # TODO(GAN): In second self.session.run, evaluate self.generator_training,
        # self.generator_summary and self.generator_loss using
        # noise sampled with `self.sample_z` as `self.z`.
        gen_loss = self.session.run(
            [self.generator_training, self.generator_summary, self.generator_loss],
            {self.images: images, self.z: self.sample_z(images.shape[0])}
        )[-1]

        # TODO(GAN): Return the sum of evaluated self.discriminator_loss and self.generator_loss.
        return gen_loss + disc_loss

    def generate(self):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.session.run(self.generated_images, {self.z: self.sample_z(GRID * GRID)})

        # Generate GRIDxGRID interpolated images
        if self.z_dim == 2:
            # Use 2D grid for sampled Z
            starts = np.stack([-np.ones(GRID), np.linspace(-1, 1, GRID)], -1)
            ends = np.stack([np.ones(GRID), np.linspace(-1, 1, GRID)], -1)
        else:
            # Generate random Z
            starts, ends = self.sample_z(GRID), self.sample_z(GRID)
        interpolated_z = []
        for i in range(GRID):
            interpolated_z.extend(starts[i] + (ends[i] - starts[i]) * np.expand_dims(np.linspace(0, 1, GRID), -1))
        interpolated_images = self.session.run(self.generated_images, {self.z: interpolated_z})

        # Stack the random images, then an empty row, and finally interpolated imates
        image = np.concatenate(
            [np.concatenate(list(images), axis=1) for images in np.split(random_images, GRID)] +
            [np.zeros([self.HEIGHT, self.WIDTH * GRID, 1])] +
            [np.concatenate(list(images), axis=1) for images in np.split(interpolated_images, GRID)], axis=0)
        self.session.run(self.generated_image_summary, {self.generated_image_data: image})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist-data", type=str, help="Dataset [fashion|cifar-cars|mnist-data].")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--recodex_validation_size", default=None, type=int, help="Validation size in ReCodEx mode.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    if args.recodex:
        data = mnist.input_data.read_data_sets(".", reshape=False, validation_size=args.recodex_validation_size, seed=42)
    elif args.dataset == "fashion":
        data = mnist.input_data.read_data_sets("fashion", reshape=False, validation_size=0, seed=42,
                                               source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")
    elif args.dataset == "cifar-cars":
        data = mnist.input_data.read_data_sets("cifar-cars", reshape=False, validation_size=0, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/cifar-cars/")
    else:
        data = mnist.input_data.read_data_sets(args.dataset, reshape=False, validation_size=0, seed=42)


    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        loss = 0
        while data.train.epochs_completed == i:
            images, _ = data.train.next_batch(args.batch_size)
            loss += network.train(images)
        print("{:.2f}".format(loss))

        network.generate()

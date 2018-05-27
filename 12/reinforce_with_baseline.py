#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            # Input states
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            # Chosen actions (used for training)
            self.actions = tf.placeholder(tf.int32, [None])
            # Observed returns (used for training)
            self.returns = tf.placeholder(tf.float32, [None])

            # Compute the action logits

            # TODO(reinforce): Add a fully connected layer processing self.states, with args.hidden_layer neurons
            # and some non-linear activatin.
            compute_layer = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)            

            # TODO(reinforce): Compute `logits` using another dense layer with
            # `num_actions` outputs (utilizing no activation function).
            logits = tf.layers.dense(compute_layer, num_actions)

            # TODO(reinforce): Compute the `self.probabilities` from the `logits`.
            self.probabilities = tf.nn.softmax(logits)

            # TODO: Compute `baseline`, by starting with a fully connected layer processing `self.states` into
            # args.hidden_layer outputs using some non-linear activation, and then employing another
            # densely connected layer with one output and no activation. Modify the result to have
            # shape `[batch_size]` (you can use for example `[:, 0]`, see the overloaded `Tensor.__getitem__` method).
            baseline_compute_layer = tf.layers.dense(self.states, args.hidden_layer_baseline, activation=tf.nn.relu)
            baseline_return = tf.layers.dense(baseline_compute_layer, 1, activation=tf.nn.relu)[:, 0]

            # Training

            # TODO: Compute final `loss` as a sum of the two following losses:
            # - softmax cross entropy loss of self.actions and `logits`.
            #   Because this is REINFORCE with a baseline, you need to weight the loss of
            #   each batch element by a difference of `self.returns` and `baseline`.
            #   Also, the gradient to `baseline` should not be propagated through this loss,
            #   so you should use `tf.stop_gradient(baseline)`.
            # - mean square error of the `self.returns` and `baseline`.
            adj_returns = tf.subtract(self.returns, tf.stop_gradient(baseline_return))
            
            reinforce_loss = tf.losses.sparse_softmax_cross_entropy(self.actions, logits, weights=adj_returns)
            baseline_loss = tf.losses.mean_squared_error(self.returns, baseline_return)
            
            loss = tf.add(baseline_loss, reinforce_loss)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.probabilities, {self.states: states})[0]

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    #np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.9999, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=512, type=int, help="Size of hidden layer.")
    parser.add_argument("--hidden_layer_baseline", default=512, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.015, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)
    
    reward_history = []
    reward_threshold = 490
    reward_threshold_train = 480
    reward_mean_length = 100


    def try_render_episode(env, args):
        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
            env.render()

    evaluating = False
    while True:
        evaluating = env.episode >= args.episodes

        # Train for a batch of episodes
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            episode_reward = 0
            state = env.reset(evaluating)
            states, actions, rewards, done = [], [], [], False
            while not done:
                try_render_episode(env, args)

                # Compute action distribution using the network
                actions_distrib = network.predict([state])

                # Choose action based on predicted distribution for current state                
                action = np.random.choice(env.actions, p=actions_distrib)
                
                # Perform one step
                next_state, reward, done, _ = env.step(action)

                # Accumulate states, actions and rewards for current episode.
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                episode_reward += reward
                state = next_state

            # Compute returns from rewards (by summing them up and applying discount by `args.gamma`).
            accumul_reward = 0
            returns = np.zeros_like(rewards)
            for i in range(1, len(returns)+1):
                accumul_reward = accumul_reward * args.gamma + rewards[-i]
                returns[-i] = accumul_reward

            # Record episode run to current batch
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

            # Record episode results
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-reward_mean_length:])

        # Stop learning if already ready to run -> start evaluating
        if avg_reward > reward_threshold and not evaluating:
            evaluating = True

        # Perform network training using recorded batched episodes & their returns.
        if not evaluating or avg_reward < reward_threshold_train:
            network.train(batch_states, batch_actions, batch_returns)

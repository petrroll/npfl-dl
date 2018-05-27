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

            # TODO: Add a fully connected layer processing self.states, with args.hidden_layer neurons
            # and some non-linear activatin.
            compute_layer = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)            

            # TODO: Compute `logits` using another dense layer with
            # `num_actions` outputs (utilizing no activation function).
            logits = tf.layers.dense(compute_layer, num_actions)

            # TODO: Compute the `self.probabilities` from the `logits`.
            self.probabilities = tf.nn.softmax(logits)

            # Training

            # TODO: Compute `loss`, as a softmax cross entropy loss of self.actions and `logits`.
            # Because this is a REINFORCE algorithm, it is crucial to weight the loss of batch
            # elements using `self.returns` -- this can be accomplished using the `weights` parameter.
            loss = tf.losses.sparse_softmax_cross_entropy(self.actions, logits, weights=self.returns)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        # We're predicting only one action for one state -> the batch is always of length 1
        return self.session.run(self.probabilities, {self.states: states})[0]

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=256, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--debug", default=False, type=bool, help="Enable debug outputs.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    reward_history = []
    reward_threshold = 490
    reward_mean_length = 100

    episodes_since_reset = 0

    def try_render_episode(env, args):
        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
            env.render()


    evaluating = False
    while True:

        # Train for a batch of episodes
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            
            # Perform episode
            episodes_since_reset += 1
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
            if args.debug: print("EVAL")
            evaluating = True

        # Reset if stuck for too long
        if episodes_since_reset > args.episodes and not evaluating:
            if args.debug: print("RESET")
            network = Network(threads=args.threads)
            network.construct(args, env.state_shape, env.actions)

            episodes_since_reset = 0

        # Perform network training using recorded batched episodes & their returns.
        if not evaluating:
            network.train(batch_states, batch_actions, batch_returns)

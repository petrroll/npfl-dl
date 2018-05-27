#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import mountain_car_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads,
                                                                       device_count = {'GPU': 0, 'CPU': 1}
))

    def construct(self, args, num_states, num_actions):
        with self.session.graph.as_default():
            # Input states
            self.states = tf.placeholder(tf.int32, [None])
            # Input q_values (uses as targets for training)
            self.q_values = tf.placeholder(tf.float32, [None, num_actions])

            # TODO: Compute one-hot representation of self.states.
            one_hot_state = tf.one_hot(self.states, num_states)

            # TODO: Compute the q_values as a single fully connected layer without activation,
            # with `num_actions` outputs, using the one-hot encoded states. It is important
            # to use such trivial architecture for the network to train at all.
            self.predictions = tf.layers.dense(one_hot_state, num_actions, activation=None)

            # Training
            # TODO: Perform the training, using mean squared error of the given
            # `q_values` and the predicted ones.
            self.loss = tf.losses.mean_squared_error(self.q_values, self.predictions)
            
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        # We don't do batches, return the first implicit one
        return self.session.run(self.predictions, {self.states: states})[0] 

    def train(self, states, q_values):
        _, loss = self.session.run((self.training, self.loss), {self.states: states, self.q_values: q_values})
        return loss

if __name__ == "__main__":
    # Fix random seed
    #np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=750, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=0.03, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--debug", default=False, type=bool, help="Enable debug outputs.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(discrete=True)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.states, env.actions)

    epsilon = args.epsilon

    alpha = args.learning_rate

    reward_history = []
    reward_threshold = -170
    reward_mean_length = 150

    episodes_since_reset = 0

    def try_render_episode(env, args):
        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
            env.render()

    # Predict q_values for current state and select best action using epsilon-greedy policy
    # ..don't use eps-greedy when evaluating.
    def get_new_state_and_q_values(network, epsilon, state, evaluating):
        state_q = network.predict([state])
        action = np.random.choice(env.actions) if np.random.random() < epsilon and not evaluating else np.argmax(state_q)
        
        return (action, state_q)
    
    evaluating = False
    while True:
        # Perform episode
        episodes_since_reset += 1
        episode_reward = 0
        state, done = env.reset(evaluating), False
        while not done:
            try_render_episode(env, args)

            # Compute q_values for current state and select new action.
            action, state_q = get_new_state_and_q_values(network, epsilon, state, evaluating)
            next_state, reward, done, _ = env.step(action)
  
            # Perform the network update (while for some tasks updating during evaluation might be
            # ..beneficial, it's not for this one (it still passes recodex but doesn't seem to 
            # ..be as stable)).
            if not evaluating:
                 # Compute the q_values of the next_state
                next_state_q = network.predict([next_state])

                # Update the (goal) q_value for the current `state` and `action` using the TD update
                # and `next_state_q` (leaving the q_values for not-taken actions unchanged).
                # |- non zero gamma seems to be very important -> makes convergence way more stable
                # |- clipping is less important with lower gamma, but still good
                state_change = reward +  (args.gamma * np.max(next_state_q) - state_q[action])
                state_change_clipped = np.clip(state_change, -1, 1)
                state_q[action] += alpha*(state_change_clipped) 

                # Train the network using the computed goal q_values for current `state`.
                network.train([state], [state_q])

            episode_reward += reward
            state = next_state
        
        # Record session results
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-reward_mean_length:])

        # Stop learning if already ready to run -> start evaluating
        if avg_reward > reward_threshold and not evaluating:
            if args.debug: print("EVAL")
            evaluating = True

        # Reset if stuck for too long
        if episodes_since_reset > args.episodes:
            if args.debug: print("RESET")
            network = Network(threads=args.threads)
            network.construct(args, env.states, env.actions)

            episodes_since_reset = 0

        # Epsilon interpolation
        if args.epsilon_final:
            epsilon = np.exp(np.interp(episodes_since_reset, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))

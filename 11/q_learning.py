#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator

def try_render_debug(env, args):
    if args.render_each and env.episode and env.episode % args.render_each == 0:
        env.render()

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    alpha = args.alpha
    eps = args.epsilon

    Q = np.zeros((env.states, env.actions))

    reward_history = []
    reward_threshold = -120
    worst_reward = -1000

    reset_after_episoded = 5000
    episodes_since_reset = 0

    # Either hard follow policy (take the best action) or with certain set p (eps) choose randomly.
    def get_action(state):
        return np.random.choice(env.actions) if np.random.random() < eps else np.argmax(Q[state, :])

    training = True
    while training:
        episodes_since_reset += 1

        episode_reward = 0
        state, done = env.reset(), False
        while not done:
            try_render_debug(env, args)
            
            # Get new action and play one step of simluation
            action = get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Update the state-action value function for the original state and selected action with
            # ..reward and the potential improvement of new state in comparison to the original one.
            Q[state, action] += alpha*( reward + (np.max(Q[next_state, :]) - Q[state, action]) ) 

            episode_reward += reward
            state = next_state

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-400:])

        closeness_to_goal = (avg_reward - worst_reward) / (-worst_reward)
        eps = args.epsilon_final + (1 - closeness_to_goal) * (args.epsilon - args.epsilon_final)
        alpha = args.alpha_final + (1 - closeness_to_goal) * (args.alpha - args.alpha_final)

        if avg_reward > reward_threshold:
            training = False

        if episodes_since_reset > reset_after_episoded:
            episodes_since_reset = 0
            Q = np.zeros((env.states, env.actions))        

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
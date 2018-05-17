#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator as cpe


def try_render_debug(env, args):
    if args.render_each and env.episode and env.episode % args.render_each == 0:
        env.render()

if __name__ == "__main__":
    # Fix random seed
    #np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.005, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # stuff for my own learning control
    rewards_history = []
    target_reward = 500
    treshold_reward = 485

    # Create the environment
    env = cpe.environment()

    # init args
    eps = args.epsilon
    gamma = args.gamma

    # init policy
    policy = np.zeros((env.states, env.actions)) + 1/env.actions

    # Combines Q and Return from Alg, remembers average return for this (state, act) combination 
    avgReturn = np.zeros((env.states, env.actions), dtype=float)
    stateActionSeen = np.zeros((env.states, env.actions), dtype=int)

    # Training 
    training = True
    while training:

        #Reset environment
        episode = []
        episode_reward = 0
        state, done = env.reset(), False

        # Record one episode
        while not done:
            try_render_debug(env, args)

            # figure out current action and move simulation
            action = np.random.choice(env.actions, 1, replace=False, p=policy[state,:])[0] # np.<..> returns array of results
            next_state, reward, done, _ = env.step(action)
            
            # append episode data & set next_state as current state
            episode.append((state, action, reward))
            episode_reward += reward
            state = next_state
        
        # update custom control variables
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-400:])
        eps = args.epsilon - (avg_reward / target_reward) * (args.epsilon - args.epsilon_final)

        # precompute certain values
        best_action_p = 1 - eps + (eps / env.actions)
        others_actions_p = (eps / env.actions)

        if (avg_reward > treshold_reward):
            training = False

        # Learn from the just recorded episode
        G = 0
        for (state, action, reward) in reversed(episode):

            # Cumulates reward that these actions will get later in this episode
            G += reward

            # Compute the average reward for current state & action combination.
            # .. and update number of elements of elements the avg. is made of.
            avgReturnCoef = stateActionSeen[state, action]
            avgReturn[state, action] = ((avgReturn[state, action] * avgReturnCoef) + G) / (avgReturnCoef + 1)
            stateActionSeen[state, action] += 1

            # find the action with highest average return
            actWithHighestRew = np.argmax(avgReturn[state,:])
            for actionIndex in range(env.actions):
                # update policy -> action with highest return (above) gets _static_ high probability all others low
                newActionP = best_action_p if actionIndex == actWithHighestRew else others_actions_p        
                policy[state, actionIndex] = newActionP

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = np.random.choice(env.actions, 1, replace=False, p=policy[state,:])[0] # np.<..> returns array of results
            next_state, reward, done, _ = env.step(action)
            state = next_state

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        # count each transition from state s to state s_next when taking action a
        self.transitionCounts = np.zeros((n_states, n_actions, n_states))
        # store the sum of rewards obtain from taking action a in state s and ending in state s_next
        self.rewardSum = np.zeros((n_states, n_actions, n_states))
        # The estimate transiton probability for taking action a in state s and ending in state s_next
        self.transitionEstimate = np.zeros((n_states, n_actions, n_states))
        # The estimate reward for taking action a in state s and ending in state s_next
        self.rewardEstimate = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        # implement epsilon-greedy action selection
        random = np.random.rand()
        if random < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s, :])
        return a

    def updateModel(self, s_begin, action, obtainedReward, s_next):
        self.transitionCounts[s_begin, action, s_next] += 1
        self.rewardSum[s_begin, action, s_next] += obtainedReward
        # Calculate the proportion of times that the agent has observed this transition
        self.transitionEstimate[s_begin,action,s_next] = self.transitionCounts[s_begin, action, s_next] / (np.sum(self.transitionCounts[s_begin, action, :]))
        # print(self.transitionCounts[s_begin, action, s_next])
        # print(np.sum(self.transitionCounts[s_begin, action, :]))
        # print(self.transitionEstimate[s_begin,action,s_next])

        # exit()
        # calculate the estimated reward for this triplet
        self.rewardEstimate = self.rewardSum[
            s_begin, action, s_next] / self.transitionCounts[s_begin, action, s_next]

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Update the model.
        self.updateModel(s, a, r, s_next)
        print(self.transitionCounts[s, a, s_next] )
    
        # Update Q-table.
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * \
            (r + self.gamma * np.max(self.Q_sa[s_next, :]) - self.Q_sa[s, a])
        # Update Q-table with playouts using the model.
        for i in range(n_planning_updates):
            # select random previously observed state using slef.transitioncounts.
            # create set of states that have been visited.
            observed_states = np.nonzero(self.transitionCounts[s,:,:])[0]
            pickedState = np.random.choice(observed_states)

            # select action from column zero in obwerved_actions.
            observedActions = []
            for action in range(self.n_actions):
                if (np.nonzero(self.transitionEstimate[s][action][:])[0] != 0):
                    observedActions.append(action)
            
            # print("action 0 taken ", (np.nonzero(self.transitionEstimate[s][0][:])[0]))
            # print("action 1 taken ", (np.nonzero(self.transitionEstimate[s][1][:])[0]))
            # print("action 2 taken ", (np.nonzero(self.transitionEstimate[s][2][:])[0]))
            # print("action 3 taken ", (np.nonzero(self.transitionEstimate[s][3][:])[0]))
            # print(observedActions)
            pickedAction = np.random.choice(observedActions)

            # obtain the next state and reward from the model.
            print(self.n_states)
            print(self.transitionEstimate[pickedState,pickedAction, :])
            s_next = np.random.choice(np.arange(self.n_states), p=self.transitionEstimate[pickedState,pickedAction, :])
            r = self.rewardEstimate[pickedState, pickedAction, s_next]
            self.Q_sa[pickedState, pickedAction] = self.Q_sa[pickedState, pickedAction] + self.learning_rate * \
                (r + self.gamma *
                 np.max(self.Q_sa[s_next, :]) - self.Q_sa[pickedState, pickedAction])
        pass


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # count each transition from state s to state s_next when taking action a
        self.transitionCounts = np.zeros((n_states, n_actions, n_states))
        # store the sum of rewards obtain from taking action a in state s and ending in state s_next
        self.rewardSum = np.zeros((n_states, n_actions, n_states))
        # The estimate transiton probability for taking action a in state s and ending in state s_next
        self.transitionEstimate = np.zeros((n_states, n_actions, n_states))
        # The estimate reward for taking action a in state s and ending in state s_next
        self.rewardEstimate = np.zeros((n_states, n_actions, n_states))

        self.Q_sa = np.zeros((n_states, n_actions))
        # TO DO: Initialize count tables, and reward sum tables.

    def select_action(self, s, epsilon):
        # TO DO: Add own code
        # Replace this with correct action selection
        a = np.random.randint(0, self.n_actions)
        return a

    def updateModel(self, s_begin, action, obtainedReward, s_next):
        self.transitionCounts[s_begin, action, s_next] += 1
        self.rewardSum[s_begin, action, s_next] += obtainedReward
        # Calculate the proportion of times that the agent has observed this transition
        self.transitionEstimate = self.transitionCounts[s_begin, action, s_next] / (np.sum(self.transitionCounts[s_begin, action, :]))
        # calculate the estimated reward for this triplet
        self.rewardEstimate = self.rewardSum(
            s_begin, action, s_next) / self.transitionCounts(s_begin, action, s_next)

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # TO DO: Add own code

        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a)))
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue
        pass


def test():

    n_timesteps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'dyna'  # 'ps'
    epsilon = 0.1
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states, env.n_actions,
                       learning_rate, gamma)  # Initialize Dyna policy
    elif policy == 'ps':
        pi = PrioritizedSweepingAgent(
            env.n_states, env.n_actions, learning_rate, gamma)  # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = False

    for t in range(n_timesteps):
        # Select action, transition, update policy
        a = pi.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        pi.update(s=s, a=a, r=r, done=done, s_next=s_next,
                  n_planning_updates=n_planning_updates)

        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)

        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input(
                "Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next


if __name__ == '__main__':
    test()

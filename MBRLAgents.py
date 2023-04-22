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
        self.transitionEstimate[s_begin, action, s_next] = self.transitionCounts[s_begin,
                                                                                 action, s_next] / (np.sum(self.transitionCounts[s_begin, action, :]))

        # calculate the estimated reward for this triplet
        self.rewardEstimate[s_begin, action, s_next] = self.rewardSum[
            s_begin, action, s_next] / self.transitionCounts[s_begin, action, s_next]

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Update the model.
        self.updateModel(s, a, r, s_next)

        # Update Q-table.
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * \
            (r + self.gamma * np.max(self.Q_sa[s_next, :]) - self.Q_sa[s, a])
        # Update Q-table with playouts using the model.
        for i in range(n_planning_updates):
            # select random previously observed state using slef.transitioncounts.
            # create set of states that have been visited.
            # np.zero returns tuple of arrays one for each dimension
            # so we look at index the observed states we look at index 0
            observed_states = np.unique(np.nonzero(
                self.transitionCounts[:, :, :])[0])

            pickedState = np.random.choice(observed_states)

            # select action that have been observed from pickedState.
            observedActions = np.nonzero(
                self.transitionCounts[pickedState, :, :])[0]
            pickedAction = np.random.choice(observedActions)
            pickedNextState = np.random.choice(np.arange(
                self.n_states), p=self.transitionEstimate[pickedState, pickedAction, :])

            r = self.rewardEstimate[pickedState, pickedAction, pickedNextState]

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
        self.pq = PriorityQueue()
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
        self.transitionEstimate[s_begin, action, s_next] = self.transitionCounts[
            s_begin, action, s_next] / (np.sum(self.transitionCounts[s_begin, action, :]))
        # calculate the estimated reward for this triplet
        self.rewardEstimate[s_begin, action, s_next] = self.rewardSum[
            s_begin, action, s_next] / self.transitionCounts[s_begin, action, s_next]

    def update(self, s, a, r, done, s_next, n_planning_updates):
        self.updateModel(s, a, r, s_next)
        p = np.abs(r + self.gamma *
                   np.max(self.Q_sa[s_next, :]) - self.Q_sa[s, a])
        # Add the state-action pair to the priority queue if its priority is above the cutoff.
        if p > self.priority_cutoff:
            self.pq.put((-p, (s, a)))

        for _ in range(n_planning_updates):
            # If the queue is empty, stop planning.
            if self.pq.empty():
                break
            # Pop the highest priority state-action pair off the queue.
            _, (priorityState, priorityAction) = self.pq.get()
            td_error = 0

            # Calculate the td (for all possible next states, given state and action)
            for s_next_prime in range(self.n_states):
                td_error += self.transitionEstimate[priorityState, priorityAction, s_next_prime] * (
                    self.rewardEstimate[priorityState, priorityAction, s_next_prime] + self.gamma * np.max(self.Q_sa[s_next_prime, :]))

            # Update model using the td calculated above.
            self.Q_sa[priorityState, priorityAction] += self.learning_rate * \
                (td_error - self.Q_sa[priorityState, priorityAction])

            # Get states that can result in ending in priorityState.
            startingStates = np.unique(np.nonzero(
                self.transitionCounts[:, :, priorityState])[0])

            for startState in startingStates:
                # Get actions that can result in ending in priorityState from startState
                observedActions = np.unique(np.nonzero(
                    self.transitionCounts[startState, :, priorityState])[0])
                for observedAction in observedActions:
                    estimatedReward = self.rewardEstimate[startState,
                                                          observedAction, priorityState]
                    
                    p = np.abs(estimatedReward + self.gamma *
                               np.max(self.Q_sa[s_next, :]) - self.Q_sa[startState, observedAction])
                    
                    # Add the state-action pair to the priority queue if its priority is above the cutoff.
                    if p > self.priority_cutoff:
                        self.pq.put((-p, (startState, observedAction)))


def test():

    n_timesteps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'ps'  # 'ps'
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

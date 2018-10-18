# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # individual Counters for each k
        # have a list/dic to store Counters for each k

        # states in tuple format (x,y)
        allStates = mdp.getStates()

        counters = []

        for i in range(iterations+1):
            if i == 0:
                counters.append(util.Counter())
            else:
                newCounter = util.Counter()
                for state in allStates:
                    actions = mdp.getPossibleActions(state)
                    expVals = []
                    if mdp.isTerminal(state):
                        newCounter[state] = 0
                    else:
                        for action in actions:
                            nextStatesProbs = mdp.getTransitionStatesAndProbs(state, action)
                            expVal = 0
                            for nextState, prob in nextStatesProbs:
                                reward = mdp.getReward(state, action, nextState)
                                # expVal += (reward + discount*counters[i-1][nextState])*prob
                                expVal += (reward + discount*self.values[nextState])*prob

                            # print(expVal)
                            expVals.append(expVal)
                        newCounter[state] = max(expVals)
                        print(newCounter)
                    counters.append(newCounter.copy())
                self.values = newCounter.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # prob of an action to a certain state * value of that certain state
        nextStatesProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        return self.getValue(nextStatesProbs[0][0])*nextStatesProbs[0][1]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions = self.mdp.getPossibleActions(state)
        rewards = []
        if not actions:
            return None
        for action in actions:
            nextStatesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            rewards.append(self.getValue(nextStatesProbs[0]))
        maxV = max(rewards)

        return actions[rewards.index(maxV)]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

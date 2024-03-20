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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Setting a default value of 0 to all the states in mdp
        self.values = {s: 0 for s in self.mdp.getStates()}
    
        # Initializing an iteration counter to 0
        iteration_count = 0

        # Looping through the iterations
        while(iteration_count < self.iterations):

            # Initializing a dictionary to store new values
            newValues = self.values.copy()

            # Getting a list of all mdp states
            states = self.mdp.getStates()

            # Looping through the state list to compute new q values
            for state in states:

                # Initializing a dictionary to store action, value pairs
                values = util.Counter()

                # Case for states other than terminal states
                if self.mdp.isTerminal(state) == False:

                    # Getting all possible actions from current state
                    actions = self.mdp.getPossibleActions(state)

                    # Looping through the list of all actions
                    for a in actions:

                        # Storing the q values in a list for each action
                        values[a] = self.getQValue(state, a)

                    # Calculating the best or maximum q value to get the best possible action
                    newValues[state] = values[values.argMax()]

            # Assigning the new qvalues to the qvalues dictionary
            self.values = newValues.copy()

            # Incrementing the counter for next iteration
            iteration_count += 1
            


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

        # Initializing the qvalue with 0
        qVal = 0

        # Looping through the transition states
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):

            # Updating the qvalue based on the value iteration function
            qVal = qVal + prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))

        # Returning the new qvalue
        return qVal
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # Case when we have reached states other than terminal state
        if self.mdp.isTerminal(state) == False:

            # Initializing a dictionary to store action, value pairs
            qValues = util.Counter()

            # Looping through the possible actions to compute the best action based on qvalues
            for action in self.mdp.getPossibleActions(state):

                # Storing the qvalues in the qvalue dictionary
                qValues[action] = self.getQValue(state, action)

            # Returning the maximum or best action
            return qValues.argMax()
        else:

            # Case when we have reached the terminal state, return No action or None
            return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Setting a default value of 0 to all the states in mdp
        self.values = {s: 0 for s in self.mdp.getStates()}


        # Initializing an iteration counter to 0
        iteration_count = 0

        # Looping through the iterations
        while(iteration_count < self.iterations):

            # Initializing a dictionary to store new values
            newValues = self.values.copy()

            # Getting a list of all mdp states
            states = self.mdp.getStates()

            # Setting the index back to the state value based on the itertation count
            index = iteration_count%len(states)

            # Initializing a dictionary to store action, value pairs
            values = util.Counter()

            # Case for states other than terminal states
            if self.mdp.isTerminal(states[index]) == False:

                # Getting all possible actions from current state
                actions = self.mdp.getPossibleActions(states[index])

                # Looping through the list of all actions
                for a in actions:

                    # Storing the q values in a list for each action
                    values[a] = self.getQValue(states[index], a)

                # Calculating the best or maximum q value to get the best possible action
                newValues[states[index]] = values[values.argMax()]

            # Assigning the new qvalues to the qvalues dictionary
            self.values = newValues.copy()

            # Incrementing the counter for next iteration
            iteration_count += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"


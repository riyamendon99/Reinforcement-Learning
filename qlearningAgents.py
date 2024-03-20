# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        # Initializing a dictionary of state, action and qvalues
        self.qvalues = util.Counter()
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # Returning the q values from the qvalue dictionary
        return self.qvalues[state, action]

        #return self.computeValueFromQValues(state)
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # Initializing an empty list for finding the max value
        maxList = []

        # Getting all the legal actions from the current state
        legalActions = self.getLegalActions(state)

        # Looping through the actions to get all the q values
        for l in legalActions:

            # Appending all the q values to the list to calculate maximum
            maxList.append(self.getQValue(state, l))

        # Case when max list is not empty, returning best or max value
        if maxList:
            return max(maxList)
        else:

            # Case when max list is empty or the state is a terminal state, returning 0.0
            return 0.0
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # Initializing a dictionary of actions and values
        actionDict = util.Counter()

        # Getting all the legal actions from the current state
        legalActions = self.getLegalActions(state)

        # Looping through the actions to get all the q values
        for l in legalActions:

            # Assigning the q values to the dictionary of actions
            actionDict[l] = self.getQValue(state, l)

        # Finding the max or best action from all the available actions and returning it
        return actionDict.argMax()
                
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action

        # Getting all the legal actions from the current state
        legalActions = self.getLegalActions(state)

        # Setting action variable default to be None
        action = None

        # Initializing a dictionary of actions and values
        actionDict = util.Counter()

        # Case when leagal actions is empty, return None
        if legalActions == None:
            return None

        # When probability is epsilon, choosing a random action from legal actions
        elif util.flipCoin(self.epsilon) == True:

            # Randomly selecting an action
            action = random.choice(legalActions)
        else:

            # In other situations, finding the maximum or the best action
            for l in legalActions:

                # Updating the action dictionary based on the qvalues
                actionDict[l] = self.getQValue(state, l)

            # Finding the max or best action from all the available actions
            action = actionDict.argMax()
            
            
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # Returning the computed action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Calculating the maximum or best qvalue from the states available
        maxVal = self.getValue(nextState)

        # Updating the qvalues
        self.qvalues[state, action] = (1-self.alpha)*self.getQValue(state, action) + self.alpha*(reward + self.discount*(maxVal))
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Extracting the feature Dictionary from the getFeatures method
        featureDict = self.featExtractor.getFeatures(state, action)

        # Getting the weights from the getWeights method
        weightDict = self.getWeights()

        # Assigning a default value of 0.0 to q value
        qVal = 0.0

        # Looping through every state, action pair to calculate the updated q values
        for f in featureDict:
            # Updating qvalue using the approximate q function
            qVal = qVal + weightDict[f]*featureDict[f]

        # Returning the updated q value
        return qVal
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Extracting the feature Dictionary from the getFeatures method
        featureDict = self.featExtractor.getFeatures(state, action)

        # Getting the weights from the getWeights method
        weightDict = self.getWeights()

        # Calculating the difference term
        difference = (reward + self.discount*self.getValue(nextState))- self.getQValue(state, action)

        # Looping through every state, action pair to calculate the updated weights
        for f in featureDict:
            # Updating weight vector
            weightDict[f] = weightDict[f] + self.alpha*difference*featureDict[f]

        # Assigning the updated values to the weight vector 
        self.weights = weightDict
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

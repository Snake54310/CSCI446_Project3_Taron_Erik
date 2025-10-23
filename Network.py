import math
import numpy as np
import os
import random 
import sys
import re
import copy as cp

class Network:
    def __init__(self, model):
        self.model = model # model imported from file
        self.probs = {} # dictionary of dictionaries for absolute probabilities
        self.vars = [] # array of all variables in model
        self.varsStates = {}
        # for each state for each variable
        for var in model.nodes(): # for every variable in tree
            varcdp = model.get_cpds(var) # get given cdps
            varStates = varcdp.state_names[var] # to get possible states
            varProbDict = {} # initialize absolute probability dictionary
            varStates = []
            for state in varStates: # for all possible states
                varProbDict.update({state : -1}) # initialize as unknown, only update once
                self.varStates.append(state)
                
            self.probs.update({var : varProbDict})
            self.vars.append(var)
            self.varsStates.update({var : varStates})
            
    def getVars(self):
        return self.vars # return list of all variables in network
    
    def getProbs(self):
        return self.probs # returns all absolute probabilities
    
    def getVarProbs(self, var): # returns probabilities for all states of a variable
        return self.probs[var]
    
    def getStateProb(self, var, state): # returns probability of a specific state on a specific variable
        return (self.getVarProbs(var)[state])
    
    def getVarsStates(self): # returns all possible states for all possible variables
        return self.varsStates
    
    def getVarStates(self, var): # returns all possible states for a specific variable
        return self.varsStates[var]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
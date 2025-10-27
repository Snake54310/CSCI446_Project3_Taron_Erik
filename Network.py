import math
import numpy as np
import os
import random 
import sys
import re
import copy as cp


class Network:
    # ---------------- INSTANTIATION ------------------
    def __init__(self, model, toReport):
        self.model = model # model imported from file
        self.toReport = toReport
        self.probs = {} # dictionary of dictionaries for absolute probabilities
        self.vars = [] # array of all variables in model
        self.varsStates = {} # dictionary providing all states names for each variable
        self.varsNumStates = {} # records the number of states for each variable
        self.varParents = {} # records parents associated with each variable
        self.varNumParents = {} # records number of parents associated with each variable
        
        self.varChildren = {} # records children associated with each variable
        self.varCDPs = {} # TabularCDP datastructure for each variable
        self.varCDPsVals = {} # For each variable, stores the probabilities of each state 
        # for each combination of states for all parents. Parent order should be identical 
        # to that in self.varParents. 
        self.parentsStates = {} # redundant with varsStates, but stores ordered states for all parents of each variable
        self.parentsNumStates = {} # redundant with varsNumStates, but stores number of states for all parents of each variable
        
        # for each state for each variable
        for var in model.nodes(): # for every variable in tree
            varcdp = model.get_cpds(var) # get given cdps
            varStates = varcdp.state_names[var] # to get possible states
            varProbDict = {} # initialize absolute probability dictionary
            newVarStates = []
            for state in varStates: # for all possible states
                varProbDict.update({state : -1}) # initialize as unknown, only update once
                newVarStates.append(state)
                
            self.probs.update({var : varProbDict})
            self.vars.append(var)
            self.varsStates.update({var : newVarStates})
            self.varsNumStates.update({var : len(newVarStates)})

            self.varParents.update({var : model.get_parents(var)})
            self.varNumParents.update({var: len(model.get_parents(var))})
            
            self.varChildren.update({var : model.get_children(var)})
            self.varCDPs.update({var: varcdp})
            self.varCDPsVals.update({var : varcdp.get_values()})

        for var in model.nodes(): # for every variable in tree (again, but separate due to previous datastructure dependencies)
            parStatesDict = {}
            parNumStatesDict = {}
            for parent in self.varParents[var]:
                parStatesDict.update({parent : self.varsStates[parent]})
                parNumStatesDict.update({parent : self.varsNumStates[parent]}) 

            self.parentsStates.update({var: parStatesDict})
            self.parentsNumStates.update({var: parNumStatesDict})
        

        # Instatiation data
        '''
        print("varNumParents: ")
        print(self.varNumParents)
        print("parentsNumStates: ")
        print(self.parentsNumStates)
        print("parentsStates: ")
        print(self.parentsStates)
        print("varCDPVals: ")
        print(self.varCDPsVals)   
        print("varCDPS: ")
        print(self.varCDPs)
        print("varChildren: ")
        print(self.varChildren)
        print("varsNumStates: ")
        print(self.varsNumStates)
        print("varParents: ")
        print(self.varParents)
        print("vars:")
        print(self.vars)
        print("probs:")
        print(self.probs)
        print("varsStates:")
        print(self.varsStates)
        '''

    # ---------------- END INSTANTIATION ------------------

    # ---------------- GET METHODS ------------------
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
    
    # ---------------- END GET METHODS ------------------

    # ---------------- SET METHODS ------------------

    
    # ---------------- END SET METHODS ------------------

    # ************************** SHARED METHODS *******************************


    # ************************** END SHARED METHODS *******************************

    # ************************** VARIABLE ELIMINATION METHODS *******************************

    # ------------------------ DO VARIABLE ELIMINATION ---------------------------------

    def doVariableElim(self):

        for var in self.toReport:
            # Solve the variable with VE
            print("Variable = " + str(var))
            print("Found Probabilities: " + str(self.getVarProbs(var)))
        
        return
    
    # ------------------------ END DO VARIABLE ELIMINATION ---------------------------------
    


    # ************************** END VARIABLE ELIMINATION METHODS *******************************

    # ************************** GIBBS SAMPLING METHODS *******************************

    # ------------------------ DO GIBBS SAMPLING ---------------------------------

    def doGibbsSample(self):

        for var in self.toReport:
            # Solve the variable with GS
            print("Variable = " + str(var))
            print("Found Probabilities: " + str(self.getVarProbs(var)))
        
        return
    
    # ------------------------ END DO GIBBS SAMPLING ---------------------------------


    # ************************** END GIBBS SAMPLING METHODS *******************************
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
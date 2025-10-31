import math
import numpy as np
import os
import random 
import sys
import re
import copy as cp


class Network:
    # ---------------- INSTANTIATION ------------------
    def __init__(self, model, toReport, evidence):
        self.model = model # model imported from file
        self.toReport = toReport
        self.evidence = evidence
        self.probs = {} # dictionary of dictionaries for absolute probabilities
        self.eprobs = {} # dictionary of dictionaries for absolute probabilities based upon children
        self.tprobs = {} # dictionary of true probabilities
        self.vars = [] # array of all variables in model
        self.varsStates = {} # dictionary providing all states names for each variable
        self.varsNumStates = {} # records the number of states for each variable
        self.varParents = {} # records parents associated with each variable
        self.varNumParents = {} # records number of parents associated with each variable
        
        self.varChildren = {} # records children associated with each variable
        self.varNumChildren = {}
        
        self.varCDPs = {} # TabularCDP datastructure for each variable
        self.varCDPsVals = {} # For each variable, stores the probabilities of each state 
        # for each combination of states for all parents. Parent order should be identical 
        # to that in self.varParents. 
        self.parentsStates = {} # redundant with varsStates, but stores ordered states for all parents of each variable
        self.parentsNumStates = {} # redundant with varsNumStates, but stores number of states for all parents of each variable

        self.childrenStates = {} # redundant with varsStates, but stores ordered states for all children of each variable
        self.childrenNumStates = {} # redundant with varsNumStates, but stores number of states for all children of each variable

        self.varsWithEvidence = [] # list of all variables with evidence
        self.evidenceDict = {} # dictionary of evidence variables and associated confirmed states
        self.isEvidenceDict = {} # dictionary containing True/False values for whether evidence was provided

        self.probsStateDistributions = {} # for some child, for some state, the probability that this state being confirmed is associated with
        # this probability distribution (as provided in the CDP ordering) is this.
        self.numProbsStateDistributions = {}

        for piece in self.evidence:
            self.varsWithEvidence.append(piece[0])
            self.evidenceDict.update({piece[0] : piece[1]})
        
        # for each state for each variable
        for var in model.nodes(): # for every variable in tree
            varcdp = model.get_cpds(var) # get given cdps
            varStates = varcdp.state_names[var] # to get possible states
            varProbDict = {} # initialize absolute probability dictionary
            varEProbDict = {} # initialize absolute probability dictionary
            varTProbDict = {} # initialize absolute probability dictionary
            newVarStates = []
            for state in varStates: # for all possible states
                varProbDict.update({state : -1}) # initialize as unknown, only update once
                varEProbDict.update({state : -1}) # initialize as unknown, only update once
                varTProbDict.update({state : -1}) # initialize as unknown, only update once
                newVarStates.append(state)
                
            self.probs.update({var : varProbDict})
            self.eprobs.update({var : varEProbDict})
            self.tprobs.update({var : varTProbDict})
            self.vars.append(var)
            self.varsStates.update({var : newVarStates})
            self.varsNumStates.update({var : len(newVarStates)})

            self.varParents.update({var : model.get_parents(var)})
            self.varNumParents.update({var: len(model.get_parents(var))})
            
            self.varChildren.update({var : model.get_children(var)})
            self.varNumChildren.update({var: len(model.get_children(var))})
            
            self.varCDPs.update({var: varcdp})
            self.varCDPsVals.update({var : varcdp.get_values()})

        for var in model.nodes(): # for every variable in tree (again, but separate due to previous datastructure dependencies)
            parStatesDict = {}
            parNumStatesDict = {}
            childStatesDict = {}
            childNumStatesDict = {}
            for parent in self.varParents[var]:
                parStatesDict.update({parent : self.varsStates[parent]})
                parNumStatesDict.update({parent : self.varsNumStates[parent]}) 

            for child in self.varParents[var]:
                childStatesDict.update({child : self.varsStates[child]})
                childNumStatesDict.update({child : self.varsNumStates[child]})

            self.parentsStates.update({var: parStatesDict})
            self.parentsNumStates.update({var: parNumStatesDict})
            
            self.childrenStates.update({var: childStatesDict})
            self.childrenNumStates.update({var: childNumStatesDict})

            self.isEvidenceDict.update({var: False})

        for var in self.varsWithEvidence:
            self.isEvidenceDict[var] = True
        
        

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

        for report in self.toReport:
            # Solve the variable with VE
            self.variableElim(report)
            print("Variable = " + str(report))
            print("Found Probabilities: " + str(self.getVarProbs(report)))
        
        return
    # ------------------------ END DO VARIABLE ELIMINATION ---------------------------------
    
    # ------------------------ VARIABLE ELIMINATION ---------------------------------
    def variableElim(self, report): # returns probability distribution of variable selected
        
        for parent in self.varParents[report]: # start at top of tree, work back down to variable in question
            if (self.probs[parent] == -1): # if parent probability has not been solved for
                self.variableElim(parent)

        localDistribution = np.zeros(self.varsNumStates[report], dtype=float)

        
        if (self.probs[report] == -1): # if we have not solved for this forward probability
            stateNumber = 0
            for state in self.varsStates[report]: # get forward probability (based purely on parent evidence)           
                # print("state: " + str(state))
                self.probs[report][state] = self.computeStateProbabilityFromParents(report, state, stateNumber)
                stateNumber += 1

        # print(self.probs[report])

        for child in self.varChildren[report]: # following all parents, solve down for children. Then, solve backwards
            if (self.eprobs[parent] == -1): # if child backwards probability has not been found
                self.variableElim(child)

        if (self.eprobs[report] == -1): # if we have not solved for this backward probability
            stateNumber = 0 
            for state in self.varsStates[report]: # get backwards probability (extracted from child evidence, based on that parent evidence)           
                # print("state: " + str(state))
                self.eprobs[report][state] = self.computeStateProbabilityFromChildren(report, state, stateNumber)
                stateNumber += 1


        for parent in self.varParents[report]: # start at top of tree, work back down to variable in question
            if (self.tprobs[parent] == -1): # if parent probability has not been solved for
                self.variableElim(parent)
        
        stateNumber = 0 
        for state in self.varsStates[report]: # get backwards probability (extracted from child evidence, based on that parent evidence)           
            # print("state: " + str(state))
            self.tprobs[report][state] = self.computeTrue(report, state, stateNumber)
            stateNumber += 1
        
        return

        
    # ------------------------ END VARIABLE ELIMINATION ---------------------------------

    # ------------------------ COMPUTE STATE PROBABILITY ---------------------------------
    def computeStateProbabilityFromParents(self, var, state, stateNumber): # CURRENTLY: returns probability of state on variable with the assumption that 
        # all parents are solved for
        # used to update self.Eprobs
        
        parentsNumsStates = np.zeros(self.varNumParents[var], dtype=int) # number of possible states for each parent
        currentParStateNums = np.zeros(self.varNumParents[var], dtype=int) # current indexes of states for each parent
        numProbsToSum = 1
        parentIndex = 1
        for parent in self.varParents[var]:
            parentsNumsStates[self.varNumParents[var] - parentIndex] = self.varsNumStates[parent] # order of parent's states must be reversed to match 
            # indexing of pgmpy's cdp
            numProbsToSum = self.varsNumStates[parent] * numProbsToSum
            parentIndex += 1
            
        probsToSum = np.zeros(numProbsToSum, dtype=float)

        overflow = 0
        #print()
        # print("Var: " + var)
        # print("parents: " + str(self.varParents[var]))
        for parentsComboIndex in range(numProbsToSum):
            thisProbability = 1
            theseParentStatesProbability = 1
            
            for parentIndexI in range(self.varNumParents[var]): # get each parent state corresponding to the current index
                currentParStateNums[parentIndexI] += overflow
                if(currentParStateNums[parentIndexI] == parentsNumsStates[parentIndexI]):
                    currentParStateNums[parentIndexI] = 0
                    overflow = 1
                else:
                    overflow = 0
                    
            overflow = 1
                
            parentIndexI = 0
            #print(parentsComboIndex)
            for parent in reversed(self.varParents[var]): # multiply the probability of each parent state at the current state index
                parentStateName = self.varsStates[parent][currentParStateNums[parentIndexI]] # get state name of current parent state's index
                theseParentStatesProbability = theseParentStatesProbability * self.probs[parent][parentStateName]
                parentIndexI += 1 # -= 1
            
            thisProbability = self.varCDPsVals[var][stateNumber][parentsComboIndex] * theseParentStatesProbability
            probsToSum[parentsComboIndex] = thisProbability # probability given this set of parents

        probability = 0
        for i in range(numProbsToSum):
            probability += probsToSum[i]

        uniformProbsToSum = np.zeros(numProbsToSum, dtype=float)
        for i in range(numProbsToSum):
            uniformProbsToSum[i] = (probsToSum[i]/probability)
        
        self.probsStateDistributions.update({var: {state: probsToSum}})
        self.numProbsStateDistributions.update({var: len(probsToSum)})

        if (self.isEvidenceDict[var]): # if probability is evidence, return a certainty
            if (state == self.evidenceDict[var]):
                probability = 1.00
                return probability
            else:
                probability = 0.00
                return probability
        
        return probability

    def computeStateProbabilityFromChildren(self, var, state, stateNumber): # used to update self.Eprobs, gets parent probabilities from 
        # previous top-down absolutes and seen children probabilities
        if (self.isEvidenceDict[var]): # if probability is evidence, return a certainty
            if (state == self.evidenceDict[var]):
                probability = 1.00
                return probability
            else:
                probability = 0.00
                return probability
                
        numberOfChildren = self.varNumChildren[var]
        if (numberOfChildren == 0): # if there are no children, then the probability is equal to the calculated absolute probability based upon parents
            probability = self.probs[var][state]
            return probability

        childrenNumsStateDistributions = np.zeros(numberOfChildren, dtype=int) # number of possible states for each child
        # currentChildStateNums = np.zeros(numberOfChildren, dtype=int) # current indexes of states for each child
        childrenStatesDistributions = [] # array of probabilities of each probability distribution on each child, given the state of the child

        childrenActualProbabilities = [] # array of raw, true propabilities for each state on each child

        #numDistsToSum = 0 # total number of disributions counted across all children
        childIndex = 0 # index of current child being computed on

        
        probsOfStateOnEachChild = np.zeros(numberOfChildren, dtype=float) # probability of each state child-to-child basis
        
        
        for child in self.varChildren[var]: # get all children. You need the probability of all distributions for every child given every state.
            childrenNumsStateDistributions[childIndex] = self.numProbsStateDistributions[child] 
            # order of children's states must be reversed to match 
            # indexing of pgmpy's cdp
            # numDistsToSum += self.numProbsStateDistributions[child]
            childrenStatesDistributions.append(self.probsStateDistributions[child])
            childrenActualProbabilities.append(self.eprobs[child])

            childNumStates = self.varsNumStates[child]
            
            probsOfEachDist = np.zeros(childrenNumsStateDistributions[childIndex], dtype=float) # probability of every distribution on child

            childParentsNumsStates = self.parentsNumStates[child]
            childNumParents = self.varNumParents[child]
            
            childParentIndex = 0 # get parent number of variable as it relates to child
            while (var != self.varParents[childParentIndex]):
                childParentIndex += 1

            parentsNumStates = np.zeros(childNumParents, dtype=int)
            parentNumStatesInd = 0
            for parentNumStates in childParentsNumsStates.values():
                parentsNumStates[parentNumStatesInd] = parentNumStates
                parentNumStatesInd += 1

            for StateDistIndex in range(childrenNumsStateDistributions[childIndex]): # for every state distribution on child 
                # (associated with number of states on all of child's parents)
                parentStateAssociations = np.zeros(childNumParents, dtype=int)
                totalStateProbDistProb = 0
                numDistProbsToWeight = childNumStates
                probDistProbsOfStates = np.zeros(childNumStates, dtype=float) # should fill with identical values
                for stateIndex in range(childNumStates): # for every possible state of child
                    childStateName = self.varsStates[child][stateIndex] # retrieve string name associated with this state on child
                    probDistgivenState = childrenStatesDistributions[childIndex][childStateName][StateDistIndex] # probability of this distribution
                    # on this child, given this state
                    probStateGivenDist = self.varCDPsVals[child][stateIndex][StateDistIndex] # probability of child state given this distribution
                    # likelihoodOnStateInDist = probDistgivenState * probStateGivenDist # probability that if child has this state, it is associated with this
                    # distribution
                    # P(A n B) = P(A|B)*P(B), || B is probability this child has our state, A is probability child is member of our distribution
                    probDistAndState = probDistgivenState * self.eprobs[child][childStateName] # probability that child has this state and it is
                    # associated with this distribution
                    probDist = 0
                    if (probStateGivenDist == 0):
                        numDistProbsToWeight -= 1
                    else:
                        # P(A n B) = P(B|A)*P(A)
                        # => P(A) = P(A n B) / P(B|A) || A is probability child is member of our distribution, B is probability this child has our state
                        probDist = probDistAndState # / probStateGivenDist # probability that this child is associated with this distribution
                        # technically speaking, because this result is no longer associated with the state 
                        # of the child, every loop should return the same value. 
                    probDistProbsOfStates[stateIndex] = probDist
                    totalStateProbDistProb += probDist

                probDist = totalStateProbDistProb #/ numDistProbsToWeight # take an average to protect accuracy
                probsOfEachDist[StateDistIndex] = probDist

                # check if StateDistIndex is associated with truthfulness of the state we're checking on var
                counted = 1
                for parent in range(childNumParents):
                    counted = counted * parentsNumStates[childNumParents - parent - 1]
                    parentStateAssociations[childNumParents - parent - 1] = StateDistIndex % counted
                    
                childDistVarStateIndex = parentStateAssociations[childParentIndex]
                    
                applies = (stateNumber == childDistVarStateIndex)

                if (applies):
                    probsOfStateOnEachChild[childIndex] += probDist
                # todo: sum all probabilties associated with our variable's (one of the child's parents states, 
                # weighted by the probabilities of our variable's states
                # then, use that probability distribution to get the probability of the state we're testing on our variable.
                

            
            childIndex += 1


        '''probability = 0
        for prob in probsOfStateOnEachChild:
            probability += prob
        probability = probability/numberOfChildren'''
        totalProb = 1
        for prob in probsOfStateOnEachChild:
            totalProb = totalProb * prob
        
        probability =  totalProb # self.probs[var][state] 
        # WILL NEED TO BE NORMALIZED ACROSS ALL STATES IN LATER CODE, OUTSIDE THIS METHOD!!!
        return probability

    def computeTrueState(self, var, state, stateNumber): # Gets true probability using top-down recursion, from eprobs
        
        parentsNumsStates = np.zeros(self.varNumParents[var], dtype=int) # number of possible states for each parent
        currentParStateNums = np.zeros(self.varNumParents[var], dtype=int) # current indexes of states for each parent
        numProbsToSum = 1
        parentIndex = 1
        for parent in self.varParents[var]:
            parentsNumsStates[self.varNumParents[var] - parentIndex] = self.varsNumStates[parent] # order of parent's states must be reversed to match 
            # indexing of pgmpy's cdp
            numProbsToSum = self.varsNumStates[parent] * numProbsToSum
            parentIndex += 1
            
        probsToSum = np.zeros(numProbsToSum, dtype=float)

        overflow = 0
        #print()
        # print("Var: " + var)
        # print("parents: " + str(self.varParents[var]))
        for parentsComboIndex in range(numProbsToSum):
            thisProbability = 1
            theseParentStatesProbability = 1
            
            for parentIndexI in range(self.varNumParents[var]): # get each parent state corresponding to the current index
                currentParStateNums[parentIndexI] += overflow
                if(currentParStateNums[parentIndexI] == parentsNumsStates[parentIndexI]):
                    currentParStateNums[parentIndexI] = 0
                    overflow = 1
                else:
                    overflow = 0
                    
            overflow = 1
                
            parentIndexI = 0
            #print(parentsComboIndex)
            for parent in reversed(self.varParents[var]): # multiply the probability of each parent state at the current state index
                parentStateName = self.varsStates[parent][currentParStateNums[parentIndexI]] # get state name of current parent state's index
                theseParentStatesProbability = theseParentStatesProbability * self.tprobs[parent][parentStateName]
                parentIndexI += 1 # -= 1
            
            thisProbability = self.varCDPsVals[var][stateNumber][parentsComboIndex] * theseParentStatesProbability
            probsToSum[parentsComboIndex] = thisProbability # probability given this set of parents

        probability = 0
        for i in range(numProbsToSum):
            probability += probsToSum[i]

        uniformProbsToSum = np.zeros(numProbsToSum, dtype=float)
        for i in range(numProbsToSum):
            uniformProbsToSum[i] = (probsToSum[i]/probability)
        
        # self.probsStateDistributions.update({var: {state: probsToSum}})
        # self.numProbsStateDistributions.update({var: len(probsToSum)})

        if (self.isEvidenceDict[var]): # if probability is evidence, return a certainty
            if (state == self.evidenceDict[var]):
                probability = 1.00
                return probability
            else:
                probability = 0.00
                return probability
        
        return probability
    # ------------------------ END COMPUTE STATE PROBABILITY ---------------------------------   
    


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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
from z3 import *
from fractions import Fraction
import DATA_GEN
#import logical_formulas_4 as lf
import itertools
import sys 
import time
start_time = time.time()

data = DATA_GEN.data_gen()
X = data[0]
Y = data[1]

class NeuralNetwork():
        
    def __init__(self, X, Y):
        self.X = X.astype(int)
        self.y = Y.astype(int)
        self.learning_rate = 0.001
        self.layers = [self.X.shape[1],8,8,1]
        self.X = X
        self.y = Y
        self.yhat = np.zeros(self.y.shape)
        self.output = np.rint(self.yhat)

        np.random.seed(1)     
        self.W1 = np.random.randn(self.layers[0], self.layers[1]) 
        self.b1  =np.random.randn(self.layers[1],)
        self.W2 = np.random.randn(self.layers[1],self.layers[2]) 
        self.b2 = np.random.randn(self.layers[2],)
        self.W3 = np.random.randn(self.layers[2],self.layers[3]) 
        self.b3 = np.random.randn(self.layers[3],)

    #careful not to take log of zero in loss function 
    def catch(self, x):
      NEAR_ZERO = 0.0000000001
      return np.maximum(x, NEAR_ZERO)

    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    def forward_propagation(self):
        Z1 = self.X.dot(self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        Z3 = A2.dot(self.W3) + self.b3
        yhat = self.sigmoid(Z3)

        self.Z1 = Z1
        self.Z2 = Z2
        self.Z3 = Z3
        self.A1 = A1
        self.A2 = A2
        self.yhat = yhat
        self.output = np.rint(self.yhat)

    def sigmoid_der(self, x):
        return x * (1 - x)

    def back_propagation(self):
        dl_wrt_yhat = np.divide(1 - self.y, self.catch(1 - self.yhat)) - np.divide(self.y, self.catch(self.yhat))
        dl_wrt_sig = self.sigmoid_der(self.yhat)
        dl_wrt_z3 = dl_wrt_yhat * dl_wrt_sig
        dl_wrt_A2 = dl_wrt_z3.dot(self.W3.T)
        dl_wrt_w3 = self.A2.T.dot(dl_wrt_z3)
        dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
        dl_wrt_z2 = dl_wrt_A2 * self.sigmoid_der(self.A2)
        dl_wrt_A1 = dl_wrt_z2.dot(self.W2.T)
        dl_wrt_w2 = self.A1.T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
        dl_wrt_z1 = dl_wrt_A1 * self.sigmoid_der(self.A1)
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        self.W1 = self.W1 -  self.learning_rate * dl_wrt_w1
        self.W2 = self.W2 -  self.learning_rate * dl_wrt_w2
        self.b1 = self.b1 -  self.learning_rate * dl_wrt_b1
        self.b2 = self.b2 -  self.learning_rate * dl_wrt_b2

    def accuracy(self):
        correct = sum(self.y == self.output) 
        acc = correct / len(self.y)
        print(str(correct) + " out of " + str(len(self.y)) + " correct")
        return acc

    def predict(self, X):
        test_data_Z1 = X.dot(self.W1) + self.b1
        test_data_A1 = self.sigmoid(test_data_Z1)
        test_data_Z2 = test_data_A1.dot(self.W2) + self.b2
        test_data_A2 = self.sigmoid(test_data_Z2)
        test_data_Z3 = test_data_A2.dot(self.W3) + self.b3
        pred = self.sigmoid(test_data_Z3)
        return pred

    def priority(self):
        p = np.absolute(np.dot(self.W2, self.W3))
        return p
    

network = NeuralNetwork(X, Y)

for i in range(2000):
    network.forward_propagation()
    network.back_propagation()
network.accuracy()
priorities = network.priority()

#puts A in descedning order according to wrt
def descend(A, wrt):
    y = np.ones((1,A.shape[1]))
    for i in range(network.layers[2]):
        maxpos = np.argmax(wrt, axis=0)
        y = np.concatenate((y, A[maxpos,:]), axis=0)
        A = np.delete(A, maxpos, 0)
        wrt = np.delete(wrt, maxpos, 0)
    ORDERED_A = np.delete(y, 0, 0)
    return ORDERED_A


A = np.concatenate((np.concatenate((np.concatenate((network.b1.T,  network.W1.T), axis=1), network.W3), axis=1), priorities), axis = 1)
absW3 = np.absolute(network.W3)

ordered_table_prior = descend(A, absW3)
ordered_priorities = np.asarray([ordered_table_prior[:, -1]])
ordered_priorities = ordered_priorities.T
ordered_table = np.delete(ordered_table_prior, -1, 1)

B = np.delete(ordered_table, -1, 1)
WITHOUT_BIAS = np.delete(B, 0, 1)
absB = np.absolute(WITHOUT_BIAS)
maxs_index = absB.argmax(axis=1)

#C is B normalised
C = np.zeros(B.shape)
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        C[i,j] = B[i,j] / WITHOUT_BIAS[i,maxs_index[i]]

def find_fraction(x, K=5):
    difference = 100000
    fraction = 0
    n_j = 1
    m_j = 1
    for n in range(-K+1, K, 1):
        for m in range(-K+1, K, 1):
            if m == 0:
                continue
            y = x - Fraction(n,m)
            if (abs(y) < abs(difference)) :
                difference = y
                fraction = Fraction(n,m)
                n_j = fraction.numerator
                m_j = fraction.denominator
    return fraction, n_j, m_j


#the denominators of all the rational fractions of the weights 
denoms = np.ones((C.shape[0], C.shape[1]-1))
denoms = denoms.astype(int)
for i in range(denoms.shape[0]):
    for j in range(denoms.shape[1]):
        denoms[i,j] = find_fraction(C[i,j+1])[2]

#the weights and biases but now the weights are in rational form ish 
D = C
for i in range(C.shape[0]):
    for j in range(1,C.shape[1], 1):
        D[i,j] = find_fraction(C[i,j])[0]

LCM = np.zeros(D.shape[0])
for i in range(denoms.shape[0]):
    LCM[i] = np.lcm.reduce(denoms[i, :])

diag_LCM = np.diag(LCM)
coeff_matrix = np.dot(diag_LCM, D)

for i in range(coeff_matrix.shape[0]):
    coeff_matrix[i,0] = np.rint(coeff_matrix[i,0])

print(coeff_matrix)
with_prior = np.concatenate((coeff_matrix, ordered_priorities), axis= 1)
print(with_prior)

descending_coeff_matrix = descend(with_prior, ordered_priorities)

#confidence interval for bias value (+/- 1)
def confidenceBias(A):
    add = [True] * A.shape[0]
    extra_matrix = np.array([[0,0,0,0]])
    for i in range(A.shape[0]):
        for j in range(extra_matrix.shape[0]):
            if ((A[i,1] == extra_matrix[j,1]) and (A[i,2] == extra_matrix[j,2])) and ((A[i,0] == (extra_matrix[j,0]) + 1) or (A[i,0] == (extra_matrix[j,0])) or (A[i,0] == (extra_matrix[j,0]) - 1)):
                add[i] = False
        if (add[i] == True):
            extra_matrix = np.concatenate((extra_matrix, [A[i]]), axis=0)
    extra_matrix = np.delete(extra_matrix, 0, 0)
    return extra_matrix


coeff_matrix = confidenceBias(descending_coeff_matrix)
print(coeff_matrix)
coeff_matrix = np.delete(coeff_matrix, 3, 1)
coeff_matrix = coeff_matrix[:4, :]
print(coeff_matrix)

x1, x2, a= Ints('x1 x2 a')
P, Q, R, S = Bools('P Q R S')
x = [1, x1, x2]
s = Solver()

#does all the postive data satisfy an inequality?
def check_inequality_1(formula):
    s.push()
    truth_a = False
    s.add(Or(a == 1, a == -1 ))
    for i in range(X.shape[0]):
        if (truth_a == True):
            s.add(a == val_a)
        s.push()
        if Y[i,0] == 1: 
            s.add((a*(formula) > 0))
            s.add(x1 == int(X[i,0]))
            s.add(x2 == int(X[i,1]))
            if (s.check() == unsat):
                #print("given formula does not work, fails for input x1 = %d and x2 = %d" %(X[i,0], X[i,1]))
                s.pop()
                s.pop()
                return False, 1
            truth_a = True
            val_a = s.model()[a]
        s.pop()
    s.pop()
    return True, val_a

#does all the postive data satisfy an inequality (<=)?
def check_equality_1(formula):
    s.push()
    truth_a = False
    s.add(Or(a == 1, a == -1 ))
    for i in range(X.shape[0]):
        if (truth_a == True):
            s.add(a == val_a)
        s.push()
        if Y[i,0] == 1: 
            s.add((a*(formula) >= 0))
            s.add(x1 == int(X[i,0]))
            s.add(x2 == int(X[i,1]))
            if (s.check() == unsat):
                #print("given formula does not work, fails for input x1 = %d and x2 = %d" %(X[i,0], X[i,1]))
                s.pop()
                s.pop()
                return False, 1
            truth_a = True
            val_a = s.model()[a]
        s.pop()
    s.pop()
    return True, val_a

#does all the negative data satisfy an inequality?
def check_inequality_0(formula):
    s.push()
    truth_a = False
    s.add(Or(a == 1, a == -1 ))
    for i in range(X.shape[0]):
        if (truth_a == True):
            s.add(a == val_a)
        s.push()
        if Y[i,0] == 0: 
            s.add((a*(formula) > 0))
            s.add(x1 == int(X[i,0]))
            s.add(x2 == int(X[i,1]))
            if (s.check() == unsat):
                #print("given formula does not work, fails for input x1 = %d and x2 = %d" %(X[i,0], X[i,1]))
                s.pop()
                s.pop()
                return False, 1
            truth_a = True
            val_a = s.model()[a]
        s.pop()
    s.pop()
    return True, val_a

#does all the negative data satisfy an inequality (<=)?
def check_equality_0(formula):
    s.push()
    truth_a = False
    s.add(Or(a == 1, a == -1 ))
    for i in range(X.shape[0]):
        if (truth_a == True):
            s.add(a == val_a)
        s.push()
        if Y[i,0] == 0: 
            s.add((a*(formula) >= 0))
            s.add(x1 == int(X[i,0]))
            s.add(x2 == int(X[i,1]))
            if (s.check() == unsat):
                #print("given formula does not work, fails for input x1 = %d and x2 = %d" %(X[i,0], X[i,1]))
                s.pop()
                s.pop()
                return False, 1
            truth_a = True
            val_a = s.model()[a]
        s.pop()
    s.pop()
    return True, val_a

#returns a list of all the possible inequalities of an atomic formula
def inequality_maker(formula):
    ineqs = []
    check_ineq_1 = check_inequality_1(formula)
    #print(check_ineq_1[0])
    if (check_ineq_1[0] == True):
        ineqs.append(check_ineq_1[1] * (formula) > 0)
        return ineqs
    check_eq_1 = check_equality_1(formula)
    #print(check_eq_1[0])
    if (check_eq_1[0] == True):
        ineqs.append(check_eq_1[1] * (formula) >= 0)
        return ineqs
    check_ineq_0 = check_inequality_0(formula)
    #print(check_ineq_0[0])
    if (check_ineq_0[0] == True):
        ineqs.append(Not(check_ineq_0[1] * (formula) > 0))
        return ineqs
    check_eq_0 = check_equality_0(formula)
    #print(check_eq_0[0])
    if (check_eq_0[0] == True):
        ineqs.append(Not(check_eq_0[1] * (formula) >= 0))
        return ineqs
    return [1 * (formula) > 0, -1 * (formula) > 0, 1 * (formula) >= 0, -1 * (formula) >= 0]

#checks if a formula fits the data
def check_formula(formula):
    for i in range(X.shape[0]):
        s.push()
        if Y[i,0] == 1: 
            s.add(formula)
        if Y[i,0] == 0:
            s.add(Not(formula))
        s.add(x1 == int(X[i,0]))
        s.add(x2 == int(X[i,1]))
        if (s.check() == unsat):
            #print("given formula does not work, fails for input x1 = %d and x2 = %d" %(X[i,0], X[i,1]))
            s.pop()
            return False
        s.pop()
    return True

atomic_formulas = [0 for i in range(coeff_matrix.shape[0])]
for i in range(coeff_matrix.shape[0]):
    for j in range(coeff_matrix.shape[1]):
        atomic_formulas[i] += coeff_matrix[i,j]*x[j]

inequalities = [inequality_maker(formula) for formula in atomic_formulas] 
num = 4 - len(inequalities)
inequalities.extend([[Not(x1 == x1)] for i in range(num)])

def printing_formula(formula):
    print("Formula returned is...")
    print(formula)
    formula = simplify(formula)
    print("this simplified is")
    print(formula)
    print("--- %s seconds ---" % (time.time() - start_time))


products = list(itertools.product(inequalities[0], inequalities[1], inequalities[2], inequalities[3]))
for product in products:
    a_formula = And(product[0], product[1], product[2], product[3])
    if (check_formula(a_formula) == True):
        printing_formula(a_formula)
        sys.exit()
    a_formula = Or(product[0], product[1], product[2], product[3])
    if (check_formula(a_formula) == True):
        printing_formula(a_formula)
        sys.exit()
    
    for combo in list(itertools.combinations([0,1,2,3], 3)):
        i = [x for x in [0,1,2,3] if x not in list(combo)][0]
        a_formula = Or(And(product[combo[0]], product[combo[1]], product[combo[2]]), product[i])
        if (check_formula(a_formula) == True):
            printing_formula(a_formula)
            sys.exit()

    for combo in list(itertools.combinations([0,1,2,3], 2)):
        second_combo = [x for x in [0,1,2,3] if x not in list(combo)]
        a_formula = Or(And(product[combo[0]], product[combo[1]]), product[second_combo[0]], product[second_combo[0]])
        if (check_formula(a_formula) == True):
            printing_formula(a_formula)
            sys.exit()

print("no formula found")
print("--- %s seconds ---" % (time.time() - start_time))

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:08:49 2022

@author: fishja
"""

from NetworkInference import NetworkInference
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

n = 10
p = 0.4
G = nx.erdos_renyi_graph(n,p)
A = nx.adjacency_matrix(G)
A = A.todense()

NI = NetworkInference()
NI.set_NetworkAdjacency(A)
NI.set_T(1000)

X1 =NI.Gen_Poisson_Data()
plt.plot(X1)

NI.set_Rho(0.95)
NI.Gen_Stochastic_Gaussian(1)
X2 = NI.return_XY()
plt.figure(2)
plt.plot(X2)

NI.set_Rho(0.95)
NI.Gen_Stochastic_Poisson_Armillotta(Betas=np.array([0.4,0.3,0.35]))
X3 = NI.return_XY()
plt.figure(3)
plt.plot(X3)


NI.Gen_Logistic_Dynamics(r = 4, sigma = 0.5)
X4 = NI.return_XY()
plt.figure(4)
plt.plot(X4)

#X3_1 = X3[0:len(X3)-1,:]
#X3_2 = X3[1:,:]

#NI.set_X(X3_1)
NI.set_XY(X4)
NI.set_InferenceMethod_oCSE('GeometricKNN')
#NI.save_state()
B = NI.Estimate_Network()
TPR,FPR = NI.Compute_TPR_FPR()
print("This is the TPR and FPR: ",TPR,FPR)
#Make sure Y is a column vector...
# B = np.zeros(A.shape)
# for i in range(A.shape[0]):
#     print(i)
#     NI.set_Y(X3_2[:,[i]])
#     NI.set_Num_Shuffles_oCSE(1000)
#     NI.set_Forward_oCSE_alpha(0.1)
#     NI.set_Backward_oCSE_alpha(0.1)
#     NI.set_InferenceMethod_oCSE('Gaussian')
#     S = NI.Alternative_oCSE()
#     print(NI.return_Sinit())
#     B[i,S] = 1
# Range = np.arange(0.01,1,0.01)
# TPRs = np.zeros(len(Range))
# FPRs = np.zeros(len(Range))
# for i in range(len(Range)):
#     NI.set_Forward_oCSE_alpha(Range[i])
#     NI.set_Backward_oCSE_alpha(Range[i])
#     B = NI.Estimate_Network()
#     TPR,FPR = NI.Compute_TPR_FPR()
#     print()
#     print()
#     print("This is the TPR and FPR: ",TPR,FPR)
#     print()
#     print()
#     TPRs[i] = TPR
#     FPRs[i] = FPR

# AUC = NI.Compute_AUC(TPRs,FPRs)
# print("This is the AUC: ", AUC)
    
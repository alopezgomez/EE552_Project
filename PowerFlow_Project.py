import pandas as pd
import numpy as np
import math
import openpyxl


"Read Excel File"
busdata_excel = pd.read_excel('EE552_Project_data.xlsx', sheet_name='BusData', engine='openpyxl')
linedata_excel = pd.read_excel('EE552_Project_data.xlsx', sheet_name='LineData', engine='openpyxl')

# try to improve this maybe numbers or some better idea
busdf = pd.DataFrame(busdata_excel, columns=['Bus #','P MW','Q MVAr','Type', 'P Gen', 'V Set'])
linedatadf = pd.DataFrame(linedata_excel, columns=['From', 'To', 'R, p.u.', 'X, p.u.', 'B, p.u.'])

busnp = busdf.to_numpy()
linedatanp = linedatadf.to_numpy()


"Base values"

Sb = 100


nf = linedatanp[:, 0]  # From
nt = linedatanp[:, 1]  # To
R = linedatanp[:, 2]   # R Line
X = linedatanp[:, 3]   # X Line
B = linedatanp[:, 4]   # B Line
B = B*1j
print("B =", B)
Z = R+(X*1j)         # Z Line
Y = 1/Z              # Y Line
nbus = int(max(max(nf), max(nt)))  # number of buses in this case nbus = 12
Ybus = np.zeros((nbus, nbus), dtype=complex)      # Y matrix of zeros nbus x nbus dtype=complex to append complex later
Pset = busnp[:, 1]  # P from bus data per unit
Qset = busnp[:, 2]  # Q from bus data per unit
Type = busnp[:, 3]   # Type from bus data
Vset = busnp[:, 5]   # Vset from bus data



"Off diagonal elements"
# i = 0-16
for i in range(len(nf)):
    Ybus[int(nf[i])-1][int(nt[i])-1] = Ybus[int(nf[i])-1][int(nt[i])-1] - Y[i]   # adding -Y[i] to initial value(zero)
    Ybus[int(nt[i])-1][int(nf[i])-1] = Ybus[int(nf[i])-1][int(nt[i])-1]   # ex: Y12 = Y21

"Diagonal elements"
for j in range(0, nbus+1):  # nbus+1 = 12+1 = 13 so j max = 12
    for i in range(len(nf)):  # number of rows From
        if int(nf[i]) == j:  # if value From = j
            Ybus[j-1][j-1] = Ybus[j-1][j-1] + Y[i] + B[i]/2  # adding Y[i] to initial value (zero)
        elif int(nt[i]) == j:  # if value To = j
            Ybus[j-1][j-1] = Ybus[j-1][j-1] + Y[i] + B[i]/2  # adding Y[i] to previous value

# Y matrix rounded to 3 decimals
Yround = np.around(Ybus, decimals=3)  # rounding to 3 decimals

print("Y round = ", Yround)

# G and B matrices
G = Ybus.real
B = Ybus.imag

# UNKNOWNS
# unknown betas = 11
# unknown voltages = 7
# known voltages from vset
# unknown P = 1  slack bus
# known P's from bus data (not Pg)
# unknown Q = 1  slack bus
# known Q's from bus data

# Known active power injections denoted as Pk
# Known reactive power injections denoted as Qk
# if the bus is load Pk should be negative
# Q do not depend on the type of bus
Pk = [0.0]*nbus
Qk = [0.0]*nbus
for i in range(nbus):
    if Type[i]=='D':
        Pk[i] = -Pset[i]
    else:
        Pk[i] = Pset[i]
    #Pinjk[i] = Pgen[i] - Pload[i]
    Qk[i] = -Qset[i]



# INITIALIZATION
# voltages and angles need to be initialized before the iterations
# theta array initialized [we know theta from the slack bus should we include in this array???]
betas = np.zeros(nbus)
# voltage array initialized
V = np.ones(nbus)


# Number of buses = N = 12
# 1 slack bus
# PV buses = m-1 = 4
# PQ buses = N-m = 7
# Unknowns = N-1 = 11 angles & N-m = 7 voltages
# Implicit = N-1 = 11 = active power eqs & N-m = 7 = reactive power eqs

# Mismatch Equations

# Define number of rows of the Jacobian (number of P and Q eqs) using a Jbus vector with the indices of each bus
# P = 11 eqs (PQ + PV) upper partition of the Jacobian
# Q = 7 eqs (PQ) lower partition of the Jacobian
# I think this can be done inside de loop but it requires further analysis which way it is better.
Jbus = []

# Put the PQ and PV buses in the upper partition of the Jacobian. Include D and G buses
for i in range(nbus):
    if (Type[i] == 'D' or Type[i] == 'G'):
        Jbus.append(i)    #Choose to store the bus index, rather than number
# Lower partition, just PQ buses (loads) which are the 'D' buses.
for i in range(nbus):
    if (Type[i] == 'D'):
        Jbus.append(i)

print("Jbus =", Jbus)
print(len(Jbus))


# outer loop : iterations
itermax = 100  # max number of iterations
max_dev = 0.0001    # Convergence tolerance, per unit MW and MVAR


# update V with Vset for the generator buses and slack bus
for i in range(nbus):
    if (Type[i] == 'S') or (Type[i] == 'G'):
        V[i] = Vset[i]

# Initialize convergence records
Pmismax = []   # maximum active power mismatch
Pmismaxbus = []  # bus where the maximum active power mismatch occurs
Qmismax = []  # maximum reactive power mismatch
Qmismaxbus = []   # bus where the maximum reactive power mismatch occurs




# Loop starts here


for iter in range(itermax):

    # Calculate injections
    # Then calculate Jbus
    # For a new iteration, injections are initialized to zero.
    P = [0.0] * nbus
    Q = [0.0] * nbus

    for i in range(0, nbus):
        for j in range(0, nbus):
            P[i] += V[i] * V[j] * (G[i][j] * math.cos(betas[i] - betas[j]) + B[i][j] * math.sin(betas[i] - betas[j]))
            Q[i] += V[i] * V[j] * (G[i][j] * math.sin(betas[i] - betas[j]) - B[i][j] * math.cos(betas[i] - betas[j]))

    # Mismatch should be equal to the length of Jbus
    # Number of mismatch equations = 11 (P eqs) + 7 (Q eqs) = 18 = len(Jbus)

    Mismatch = [0.0] * (len(Jbus))   # mistmatch array equal to the length of Jbus = 18
    Pmismax.append(0.0)  # Pmismax = [0.0] do not need to be the length of Jbus because it is the max mismatch
    Qmismax.append(0.0)  # Qmismax = [0.0]
    Pmismaxbus.append(0) # Pmismaxbus = [0.0]
    Qmismaxbus.append(0) # Qmismaxbus = [0.0]

    for i in range(len(Jbus)):
        if (i < nbus - 1):
            Mismatch[i] = Pk[Jbus[i]] - P[Jbus[i]]
            if (abs(Mismatch[i]) > Pmismax[iter]):
                Pmismax[iter] = abs(Mismatch[i])
                Pmismaxbus[iter] = Jbus[i] + 1
        else:
            Mismatch[i] = Qk[Jbus[i]] - Q[Jbus[i]]
            if (abs(Mismatch[i]) > Qmismax[iter]):
                Qmismax[iter] = abs(Mismatch[i])
                Qmismaxbus[iter] = Jbus[i] + 1

    # Check for convergence

    mismax = max([abs(np.max(Mismatch)), abs(np.min(Mismatch))])
    ##    print('iter %d max mismatch %.8f' % (iter, mismax))
    if (mismax < max_dev):
        print('Converged!')
        break

    # Check for excess iterations (goes out of loop with voltage solution
    # used to find mismatch
    if (iter == itermax - 1):
        print('Too many iterations')
        break


    # Initialize Jacobian
    J = [[0.0 for i in range(len(Jbus))] for j in range(len(Jbus))]



    # Calculating the terms of the Jacobian matrix"

    # J11 top left submatrix

    for i in range(nbus-1):   # range 0 to nbus-1 = 11
        for j in range(nbus-1): # range 0 to nbus-1 = 11
            ibus = Jbus[i]   # set ibus equal to the bus numbers inside Jbus
            jbus = Jbus[j]   # set jbus equal to the bus numbers inside Jbus
            if ibus == jbus:  # Diagonal term
                J[i][j] = -Q[ibus] - (V[ibus] ** 2) * B[ibus][jbus]
            else:  # Off diagonal
                J[i][j] = V[ibus] * V[jbus] \
                            * (G[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]) \
                               - B[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]))

    # [J21] bottom left submatrix
    for i in range(nbus-1, len(Jbus)):  # range  11 to 18
        for j in range(nbus-1):   # range  0 to nbus-1 = 11
            ibus = Jbus[i]
            jbus = Jbus[j]
            if ibus == jbus:  # Diagonal term
                J[i][j] = P[ibus] - (V[ibus]**2)*G[ibus][ibus]
            else:   # Off diagonal
                J[i][j] = - V[ibus] * V[jbus] \
                 * (G[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]) \
                 +  B[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]))

    # [J12] top right submatrix
    for i in range(nbus-1): # range 0 to nbus-1 =  11
        for j in range(nbus-1,len(Jbus)): # range 11 to 18
            ibus = Jbus[i]
            jbus = Jbus[j]
            if ibus == jbus:  # Diagonal term
                J[i][j] = (P[ibus] / V[ibus]) \
                            + V[ibus] * G[ibus][ibus]
            else:   # Off diagonal
                J[i][j] = - V[ibus] \
                 * (G[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]) \
                 +  B[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]))

    # [J22] bottom right submatrix
    for i in range(nbus-1,len(Jbus)): # range 11 to 18
        for j in range(nbus-1,len(Jbus)): # range 11 to 18
            ibus = Jbus[i]
            jbus = Jbus[j]
            if ibus == jbus:  # Diagonal term
                J[i][j] = (Q[ibus] / V[ibus]) \
                            - V[ibus] * B[ibus][ibus]
            else:   # Off diagonal
                J[i][j] = V[ibus] \
                 * (G[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]) \
                 -  B[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]))


    #
    # Find the voltage correction
    #
    deltaV = np.linalg.solve(J, Mismatch)
    #
    #  Apply voltage correction
    #
    for i in range(len(Jbus)):
        if (i < nbus-1):   # in the betas
            ibus = Jbus[i]
            betas[ibus] += deltaV[i]
        else:           # in the magnitudes
            ibus = Jbus[i]
            V[ibus] += deltaV[i]



print('Convergence History')
print('Iter  P Mismatch  P Bus  Q Mismatch  Q Bus')
for i in range(len(Pmismax)):
    print('%2d %12.7f %2d %12.7f %2d' % \
         (i, Pmismax[i], Pmismaxbus[i], Qmismax[i], Qmismaxbus[i]))

print('Bus Results')
print('Bus    V     Beta      PG      PL       Pk')
for i in range(nbus):
    print('%3d %6.3f %6.4f  %6.1f   %6.1f   %6.4f' % (i+1, V[i], betas[i]*180/math.pi,
                                                       (P[i]-Pset[i]),Pset[i],P[i]))















# --------------------------------------------------------------------------------------------------
# EE 552 PROJECT CODE BY ADRIAN LOPEZ
# The code is divided in 5 sections:
# 1. READING DATA AND DEFINING SIMULATION PARAMETERS  LINES (
# 2. POWER FLOW  LINES (
# 3. SECOND-ORDER MODEL  LINES (
# 4. THIRD-ORDER MODEL   LINES (
# 5. FOURTH-ORDER MODEL  LINES (
# Values such as damping, fault clearing time and loading are defined at the beginning of the code.
# Damping is not considered in the 2nd order model (changing the damping factor will not have any effect on this model)
# Damping is considered for the 3rd order and 4rd order models
# Changes is loading and fault clearing time affect all of the models
# --------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Functions for converting complex numbers

def rectangular(module, angle):
    return cmath.rect(module, (angle * math.pi) / 180)

def polar(number):
    return (cmath.polar(number)[0], cmath.polar(number)[1] * (180 / math.pi))

def deg(angle):
    return (angle * 180) / math.pi

# 1. READING DATA AND DEFINING SIMULATION PARAMETERS

# READING SYSTEM DATA FROM EXCEL FILE

busdata_excel = pd.read_excel('EE552_Project_data.xlsx', sheet_name='BusData', engine='openpyxl')
linedata_excel = pd.read_excel('EE552_Project_data.xlsx', sheet_name='LineData', engine='openpyxl')

# try to improve this maybe numbers or some better idea
busdf = pd.DataFrame(busdata_excel, columns=['Bus #','P MW','Q MVAr','Type', 'P Gen', 'V Set'])
linedatadf = pd.DataFrame(linedata_excel, columns=['From', 'To', 'R, p.u.', 'X, p.u.', 'B, p.u.'])

busnp = busdf.to_numpy()
linedatanp = linedatadf.to_numpy()

# Base Values

Sb = 100


nf = linedatanp[:, 0]  # From
nt = linedatanp[:, 1]  # To
R = linedatanp[:, 2]   # R Line
X = linedatanp[:, 3]   # X Line
B = linedatanp[:, 4]   # B Line
B = B*1j
Z = R+(X*1j)         # Z Line
Y = 1/Z              # Y Line
nbus = int(max(max(nf), max(nt)))  # number of buses in this case nbus = 12
Ybus = np.zeros((nbus, nbus), dtype=complex)      # Y matrix of zeros nbus x nbus dtype=complex to append complex later
Pset = busnp[:, 1]  # P from bus data per unit
Qset = busnp[:, 2]  # Q from bus data per unit
Type = busnp[:, 3]   # Type from bus data
Vset = busnp[:, 5]   # Vset from bus data


# SIMULATION PARAMETERS
xd_prime = [0.08, 0.18, 0.12]
load_multiplier = [0.5, 0.5]  # new load = multiplier * previous load (how changes in load affect the dynamics)
ng = 3   # number of generators
nl = 2   # number of loads
Tdo_prime = [0.84, 0.83, 0.81] # transient d-axis time constants
xd = [1.8, 1.7, 1.5]  # d-axis steady-state reactances
time_fault = 0.1   # Fault clearing time
time_simulation = 1.5  # Simulation time
n_steps = 1000  # simulation steps
D = [0, 0, 0]    # Damping for each generator
H = np.array([10, 3.01, 6.4])  # Inertia constants
f0 = 60  # Frequency

# Increasing the loads by the multiplier
Pset[3] = Pset[3]*load_multiplier[0]
Pset[4] = Pset[4]*load_multiplier[1]
Qset[3] = Qset[3]*load_multiplier[0]
Qset[4] = Qset[4]*load_multiplier[1]

# 2. POWER FLOW

# Y-matrix

# Off-diagonal Elements
# i = 0-16
for i in range(len(nf)):
    Ybus[int(nf[i])-1][int(nt[i])-1] = Ybus[int(nf[i])-1][int(nt[i])-1] - Y[i]   # adding -Y[i] to initial value(zero)
    Ybus[int(nt[i])-1][int(nf[i])-1] = Ybus[int(nf[i])-1][int(nt[i])-1]   # ex: Y12 = Y21

# Diagonal Elements
for j in range(0, nbus+1):  # nbus+1 = 12+1 = 13 so j max = 12
    for i in range(len(nf)):  # number of rows From
        if int(nf[i]) == j:  # if value From = j
            Ybus[j-1][j-1] = Ybus[j-1][j-1] + Y[i] + B[i]/2  # adding Y[i] to initial value (zero)
        elif int(nt[i]) == j:  # if value To = j
            Ybus[j-1][j-1] = Ybus[j-1][j-1] + Y[i] + B[i]/2  # adding Y[i] to previous value

# Y matrix rounded to 3 decimals
Yround = np.around(Ybus, decimals=3)  # rounding to 3 decimals

# print("Y round = ", Yround) to check the Y matrix

# G and B matrices
G = Ybus.real
B = Ybus.imag

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
    Qk[i] = -Qset[i]


# INITIALIZATION
# voltages and angles need to be initialized before the iterations
betas = np.zeros(nbus)
V = np.ones(nbus)


# Define number of rows of the Jacobian (number of P and Q eqs) using a Jbus vector with the indices of each bus

Jbus = []

# Put the PQ and PV buses in the upper partition of the Jacobian. Include D and G buses
for i in range(nbus):
    if (Type[i] == 'D' or Type[i] == 'G'):
        Jbus.append(i)    #Choose to store the bus index, rather than number
# Lower partition, just PQ buses (loads) which are the 'D' buses.
for i in range(nbus):
    if (Type[i] == 'D'):
        Jbus.append(i)

# print("Jbus =", Jbus)
# print(len(Jbus))


# outer loop : iterations
itermax = 100  # max number of iterations
max_dev = 0.0001    # max deviation, per unit MW and MVAR


# update V with Vset for the generator buses and slack bus
for i in range(nbus):
    if (Type[i] == 'S') or (Type[i] == 'G'):
        V[i] = Vset[i]

# Initialize convergence records
Pmismax = []   # maximum active power mismatch
Pmismaxbus = []  # bus where the maximum active power mismatch occurs
Qmismax = []  # maximum reactive power mismatch
Qmismaxbus = []   # bus where the maximum reactive power mismatch occurs




# Power flow loop


for iter in range(itermax):

    # Injections are calculated
    # Then the Jacobian is calculated
    # Injections are initialized to zero
    P = [0.0] * nbus
    Q = [0.0] * nbus

    for i in range(0, nbus):
        for j in range(0, nbus):
            P[i] += V[i] * V[j] * (G[i][j] * math.cos(betas[i] - betas[j]) + B[i][j] * math.sin(betas[i] - betas[j]))
            Q[i] += V[i] * V[j] * (G[i][j] * math.sin(betas[i] - betas[j]) - B[i][j] * math.cos(betas[i] - betas[j]))

    # Mismatch should be equal to the length of Jbus

    Mismatch = [0.0] * (len(Jbus))   # mistmatch array equal to the length of Jbus
    Pmismax.append(0.0)  # Pmismax = [0.0]
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

    mismax = max([abs(np.max(Mismatch)), abs(np.min(Mismatch))])  # max absolute value of mismatch

    if (mismax < max_dev):
        print('Converged!')
        break

    # Check for excess iterations (goes out of loop with voltage solution
    # used to find mismatch
    if (iter == itermax - 1):
        print('Max number of iterations reached')
        break


    # Initialize Jacobian
    J = [[0.0 for i in range(len(Jbus))] for j in range(len(Jbus))]

    # Calculating the terms of the Jacobian matrix"

    # J11 top left submatrix

    for i in range(nbus-1):   # range 0 to nbus-1
        for j in range(nbus-1): # range 0 to nbus-1
            ibus = Jbus[i]   # set ibus equal to the bus numbers inside Jbus
            jbus = Jbus[j]   # set jbus equal to the bus numbers inside Jbus
            if ibus == jbus:  # Diagonal term
                J[i][j] = -Q[ibus] - (V[ibus] ** 2) * B[ibus][jbus]
            else:  # Off diagonal
                J[i][j] = V[ibus] * V[jbus] \
                            * (G[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]) \
                               - B[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]))

    # [J21] bottom left submatrix
    for i in range(nbus-1, len(Jbus)):
        for j in range(nbus-1):
            ibus = Jbus[i]
            jbus = Jbus[j]
            if ibus == jbus:  # Diagonal term
                J[i][j] = P[ibus] - (V[ibus]**2)*G[ibus][ibus]
            else:   # Off diagonal
                J[i][j] = - V[ibus] * V[jbus] \
                 * (G[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]) \
                 +  B[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]))

    # [J12] top right submatrix
    for i in range(nbus-1):
        for j in range(nbus-1,len(Jbus)):
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
    for i in range(nbus-1,len(Jbus)):
        for j in range(nbus-1,len(Jbus)):
            ibus = Jbus[i]
            jbus = Jbus[j]
            if ibus == jbus:  # Diagonal term
                J[i][j] = (Q[ibus] / V[ibus]) \
                            - V[ibus] * B[ibus][ibus]
            else:   # Off diagonal
                J[i][j] = V[ibus] \
                 * (G[ibus][jbus] * math.sin(betas[ibus] - betas[jbus]) \
                 -  B[ibus][jbus] * math.cos(betas[ibus] - betas[jbus]))

    # Find the corrections
    #
    delta_pf = np.linalg.solve(J, Mismatch)
    #
    #  Apply the corrections
    #
    for i in range(len(Jbus)):
        if (i < nbus-1):   # in the betas
            ibus = Jbus[i]
            betas[ibus] += delta_pf[i]
        else:           # in the magnitudes
            ibus = Jbus[i]
            V[ibus] += delta_pf[i]


print('Power Flow Results')
print('Bus    V     Beta      Pinj')
for i in range(nbus):
    print('%3d %6.3f %6.4f %6.4f' % (i+1, V[i], betas[i]*180/math.pi,
                                                       P[i]))



# 3. SECOND-ORDER MODEL

# Calculating the admittance between the generator internal bus and terminal bus.

y14 = y41 = 1/(xd_prime[0]*1j)
y25 = y52 = 1/(xd_prime[1]*1j)
y36 = y63 = 1/(xd_prime[2]*1j)


# Step 2: admittances corresponding to the loads:

y77 = -(P[7-ng-1]*load_multiplier[0]/V[7-ng-1]**2 - Q[7-ng-1]*load_multiplier[0]*1j/V[7-ng-1]**2)
y88 = -(P[8-ng-1]*load_multiplier[1]/V[8-ng-1]**2 - Q[8-ng-1]*load_multiplier[1]*1j/V[8-ng-1]**2)

print(y77)
print(y88)

# Step 3: The internal generator voltages are calculated
E = np.zeros(ng, dtype=complex)
E_0 = np.zeros(ng, dtype=complex)
delta0 = np.zeros(ng)
for i in range(0, ng):
    E[i] = V[i]+Q[i]*xd_prime[i]/V[i] + (P[i]*xd_prime[i]/V[i])*1j
    print("E_", i+1, "=",  polar(E[i]))
    delta0[i] = np.angle(E[i]) + betas[i]
    print("delta_", i+1, "=", deg(delta0[i]))
    E_0[i] = cmath.rect(abs(E[i]), delta0[i])    #use deg just for printing
    print("E_delta0 =", i+1, "=", polar(E_0[i]))

# Step 4: Prefault, Faulted, and Postfault admittance matrices

# PREFAULT MATRIX

Y_pre = np.zeros((nbus+ng, nbus+ng), dtype=complex)

# Off diagonal Elements

# Elements that remain equal to the previous matrix
for i in range(0, nbus):
    for j in range(0, nbus):
        if i != j:
            Y_pre[i+ng][j+ng] = Ybus[i][j]

# Elements that include the xd_prime
# this needs a loop

Y_pre[0][3] = Y_pre[3][0] = -y14
Y_pre[1][4] = Y_pre[4][1] = -y25
Y_pre[2][5] = Y_pre[5][2] = -y36




# Diagonal Elements
# elements that remain the same
# Y44, Y55, Y66 need to include the generator reactance


# xd_prime elements
Y_pre[0][0] = y14
Y_pre[1][1] = y25
Y_pre[2][2] = y36

# rest of the elements
Y_pre[3][3] = Ybus[0][0] + y14
Y_pre[4][4] = Ybus[1][1] + y25
Y_pre[5][5] = Ybus[2][2] + y36

# elements of the load nodes
Y_pre[6][6] = Ybus[3][3] + y77
Y_pre[7][7] = Ybus[4][4] + y88


print("Y prefault =", Y_pre)


# FAULTED MATRIX
bus_fault = 7
Y_fault = np.zeros((nbus+ng, nbus+ng), dtype=complex)

for i in range(nbus+ng):
    for j in range(nbus+ng):
            Y_fault[i][j] = Y_pre[i][j]

for i in range(nbus+ng):
    for j in range(nbus+ng):
        if i == bus_fault-1:
            Y_fault[i][j] = 0
        if j == bus_fault-1:
            Y_fault[i][j] = 0

print("Y faulted = ", Y_fault)

# POST FAULT MATRIX
# Line between buses 7-6 is removed

Y_post = np.zeros((nbus+ng, nbus+ng), dtype=complex)

for i in range(nbus+ng):
    for j in range(nbus+ng):
            Y_post[i][j] = Y_pre[i][j]

Y_post[5][5] = Y_pre[5][5] - Y[5]
Y_post[6][6] = Y_pre[6][6] - Y[5]
Y_post[5][6] = Y_pre[5][6] + Y[5]
Y_post[6][5] = Y_pre[6][5] + Y[5]

print(" Y post =", Y_post)

print(Y_pre[6][6])


# Kron reduction (method 1)

# for k in range(nbus+ng-1, ng-1, -1):
#     print(k)
#     for i in range(0, k):
#         if i !=k:
#             for j in range(0, k):
#                 if j != k:
#                     Y_pre[i][j] = Y_pre[i][j] - (Y_pre[i][k]*Y_pre[k][j])/Y_pre[k][k]
#                     Y_post[i][j] = Y_post[i][j] - (Y_post[i][k] * Y_post[k][j]) / Y_post[k][k]
#
# Y_pre_reduced = np.zeros((ng, ng), dtype=complex)
# Y_fault_reduced = np.zeros((ng, ng), dtype=complex)
# Y_post_reduced = np.zeros((ng, ng), dtype=complex)
#
# for i in range(0, ng):
#     for j in range(0, ng):
#         Y_pre_reduced[i][j] = Y_pre[i][j]
#         Y_post_reduced[i][j] = Y_post[i][j]
#
#
# print(Y_pre_reduced)


# Kron reduction (method 2 : with products of matrices)

# Y prefault matrix

Ynn1 = Y_pre[:3, :3]
Yns1 = Y_pre[:3, 3:]
Ysn1 = Y_pre[3:, :3]
Yss1 = Y_pre[3:, 3:]

Y_pre_reduced = Ynn1-np.matmul(Yns1, (np.matmul(np.linalg.inv(Yss1), Ysn1)))

print(" Y pre reduced =", Y_pre_reduced)

# Y faulted matrix
# delete the row and column with zeros

Y_fault = np.delete(Y_fault, 6, 0)
Y_fault = np.delete(Y_fault, 6, 1)

Ynn2 = Y_fault[:3, :3]
Yns2 = Y_fault[:3, 3:]
Ysn2 = Y_fault[3:, :3]
Yss2 = Y_fault[3:, 3:]

Y_fault_reduced = Ynn2-np.matmul(Yns2, (np.matmul(np.linalg.inv(Yss2), Ysn2)))

print(" Y fault reduced =", Y_fault_reduced)

# Y post-fault matrix

Ynn3 = Y_post[:3, :3]
Yns3 = Y_post[:3, 3:]
Ysn3 = Y_post[3:, :3]
Yss3 = Y_post[3:, 3:]

Y_post_reduced = Ynn3-np.matmul(Yns3, (np.matmul(np.linalg.inv(Yss3), Ysn3)))

print(" Y post reduced =", Y_post_reduced)


# differential equations
# the value of the mechanical power is equal to the electrical power at the prefault conditions
# initial angles are given by delta0

Pm = P[:3]
print(Pm)

# calculating the G and B

G_pre = Y_pre_reduced.real
B_pre = Y_pre_reduced.imag
G_fault = Y_fault_reduced.real
B_fault = Y_fault_reduced.imag
G_post = Y_post_reduced.real
B_post = Y_post_reduced.imag

# calculating M
M = H/(math.pi*f0)


# prefault numerical integration



def fault_2nd_order(x, t):
    # Assign each ODE to vector element
    delta1 = x[0]
    omega1 = x[1]
    delta2 = x[2]
    omega2 = x[3]
    delta3 = x[4]
    omega3 = x[5]

    G1 = Y_fault_reduced.real
    B1 = Y_fault_reduced.imag

    # Define each ODE

    a = omega1
    b = (Pm[0] - (abs(E[0])**2)*G1[0][0] -
         (abs(E[0]) * abs(E[1]) * (B1[0][1] * math.sin(delta1 - delta2) + G1[0][1] * math.cos(delta1 - delta2)) +
          abs(E[0]) * abs(E[2]) * (B1[0][2] * math.sin(delta1 - delta3) + G1[0][2] * math.cos(delta1 - delta3)))) / M[0]
    c = omega2
    d = (Pm[1] - (abs(E[1])**2)*G1[1][1] -
         (abs(E[1]) * abs(E[0]) * (B1[1][0] * math.sin(delta2 - delta1) + G1[1][0] * math.cos(delta2 - delta1)) +
          abs(E[1]) * abs(E[2]) * (B1[1][2] * math.sin(delta2 - delta3) + G1[1][2] * math.cos(delta2 - delta3)))) / M[1]
    e = omega3
    f = (Pm[2] - (abs(E[2])**2)*G1[2][2] -
         (abs(E[2]) * abs(E[0]) * (B1[2][0] * math.sin(delta3 - delta1) + G1[2][0] * math.cos(delta3 - delta1)) +
          abs(E[2]) * abs(E[1]) * (B1[2][1] * math.sin(delta3 - delta2) + G1[2][1] * math.cos(delta3 - delta2)))) / M[2]

    return a, b, c, d, e, f



t_fault = np.linspace(0, time_fault, n_steps)  # fault simulation time
ic_fault_2nd_order = [delta0[0], 0, delta0[1], 0, delta0[2], 0]   # initial conditions


sol_fault_2nd_order = odeint(fault_2nd_order, ic_fault_2nd_order, t_fault)

delta1_fault_2nd_order = sol_fault_2nd_order[:, 0]
omega1_fault_2nd_order = sol_fault_2nd_order[:, 1]
delta2_fault_2nd_order = sol_fault_2nd_order[:, 2]
omega2_fault_2nd_order = sol_fault_2nd_order[:, 3]
delta3_fault_2nd_order = sol_fault_2nd_order[:, 4]
omega3_fault_2nd_order = sol_fault_2nd_order[:, 5]



def postfault_2nd_order(x, t):
    # Assign each ODE to vector element
    delta1 = x[0]
    omega1 = x[1]
    delta2 = x[2]
    omega2 = x[3]
    delta3 = x[4]
    omega3 = x[5]

    G = Y_post_reduced.real
    B = Y_post_reduced.imag

    # Define each ODE

    a = omega1
    b = (Pm[0] - (abs(E[0])**2)*G[0][0] -
         (abs(E[0]) * abs(E[1]) * (B[0][1] * math.sin(delta1 - delta2) + G[0][1] * math.cos(delta1 - delta2)) +
          abs(E[0]) * abs(E[2]) * (B[0][2] * math.sin(delta1 - delta3) + G[0][2] * math.cos(delta1 - delta3)))) / M[0]
    c = omega2
    d = (Pm[1] - (abs(E[1])**2)*G[1][1] -
         (abs(E[1]) * abs(E[0]) * (B[1][0] * math.sin(delta2 - delta1) + G[1][0] * math.cos(delta2 - delta1)) +
          abs(E[1]) * abs(E[2]) * (B[1][2] * math.sin(delta2 - delta3) + G[1][2] * math.cos(delta2 - delta3)))) / M[1]
    e = omega3
    f = (Pm[2] - (abs(E[2])**2)*G[2][2] -
         (abs(E[2]) * abs(E[0]) * (B[2][0] * math.sin(delta3 - delta1) + G[2][0] * math.cos(delta3 - delta1)) +
          abs(E[2]) * abs(E[1]) * (B[2][1] * math.sin(delta3 - delta2) + G[2][1] * math.cos(delta3 - delta2)))) / M[2]

    return a, b, c, d, e, f





t_post = np.linspace(time_fault, time_simulation, n_steps)

ic_post_2nd_order = [sol_fault_2nd_order[999, 0], sol_fault_2nd_order[999, 1], sol_fault_2nd_order[999, 2],
          sol_fault_2nd_order[999, 3], sol_fault_2nd_order[999, 4], sol_fault_2nd_order[999, 5]]

sol_post_2nd_order = odeint(postfault_2nd_order, ic_post_2nd_order, t_post)

delta1_post_2nd_order = sol_post_2nd_order[:, 0]
omega1_post_2nd_order = sol_post_2nd_order[:, 1]
delta2_post_2nd_order = sol_post_2nd_order[:, 2]
omega2_post_2nd_order = sol_post_2nd_order[:, 3]
delta3_post_2nd_order = sol_post_2nd_order[:, 4]
omega3_post_2nd_order = sol_post_2nd_order[:, 5]

# plot the results
# Plot of the angles
plt.plot(t_fault, deg(delta1_fault_2nd_order), color='blue', label=r'$\delta_{1}$')
plt.plot(t_fault, deg(delta2_fault_2nd_order), color='green', label=r'$\delta_{2}$')
plt.plot(t_fault, deg(delta3_fault_2nd_order), color='red', label=r'$\delta_{3}$')
plt.plot(t_post, deg(delta1_post_2nd_order), color='blue')
plt.plot(t_post, deg(delta2_post_2nd_order), color='green')
plt.plot(t_post, deg(delta3_post_2nd_order), color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor angles in degrees')
plt.title("Rotor angles using the second order model")
plt.show()

# Plot of delta21 and delta 31
plt.plot(t_fault, deg(delta2_fault_2nd_order - delta1_fault_2nd_order), color='green', label=r'$\delta_{21}$')
plt.plot(t_fault, deg(delta3_fault_2nd_order - delta1_fault_2nd_order), color='red', label=r'$\delta_{31}$')
plt.plot(t_post, deg(delta2_post_2nd_order - delta1_post_2nd_order), color='green')
plt.plot(t_post, deg(delta3_post_2nd_order - delta1_post_2nd_order), color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor angles in degrees')
plt.title("Rotor angles with slack bus as reference using the second order model")
plt.show()


# Plot of omegas
plt.plot(t_fault, omega1_fault_2nd_order/377+1, color='blue', label=r'$\omega_{1}$')
plt.plot(t_fault, omega2_fault_2nd_order/377+1, color='green', label=r'$\omega_{2}$')
plt.plot(t_fault, omega3_fault_2nd_order/377+1, color='red', label=r'$\omega_{3}$')
plt.plot(t_post, omega1_post_2nd_order/377+1, color='blue')
plt.plot(t_post, omega2_post_2nd_order/377+1, color='green')
plt.plot(t_post, omega3_post_2nd_order/377+1, color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor speed in radians per sec/377')
plt.title("Rotor speeds using the second order model")
plt.show()




# 4. Third Order Model

# Finding Ef
# Ed0 = 0 and E'd0 = 0 as in salient pole generators

Ef = np.zeros(ng, dtype=complex)
Ef_0 = np.zeros(ng, dtype=complex)
delta_Ef = np.zeros(ng)

for i in range(0, ng):
    Ef[i] = V[i]+Q[i]*xd[i]/V[i] + (P[i]*xd[i]/V[i])*1j
    delta_Ef[i] = np.angle(Ef[i]) + betas[i]
    Ef_0[i] = cmath.rect(abs(Ef[i]), delta_Ef[i])
    print("Ef_0 =", i+1, "=", polar(Ef_0[i]))


# Finding Id

I = np.zeros(ng, dtype=complex)
I_0 = np.zeros(ng, dtype=complex)
Id = np.zeros(ng)
phi = np.zeros(ng)

for i in range(0, ng):
    I[i] = (P[i]-Q[i]*1j)/abs(V[i])
    phi[i] = np.angle(I[i]) - betas[i]
    I_0[i] = cmath.rect(abs(I[i]), phi[i])    #use deg just for printing
    print("I_0 =", i+1, "=", polar(I_0[i]))
    Id[i] = -abs(I_0[i])*math.sin(-phi[i] + delta_Ef[i])

# Finding the d-component as in example 4.2 of the Machoswki book
print("Id =", Id)



def fault_3rd_order(x, t):
    # Assign each ODE to vector element
    delta1 = x[0]
    omega1 = x[1]
    Eq1_prime = x[2]
    delta2 = x[3]
    omega2 = x[4]
    Eq2_prime = x[5]
    delta3 = x[6]
    omega3 = x[7]
    Eq3_prime = x[8]


    G1 = Y_fault_reduced.real
    B1 = Y_fault_reduced.imag

    # Define each ODE

    a = omega1
    b = (Pm[0] - D[0]*omega1 - (Eq1_prime**2)*G1[0][0] -
         (Eq1_prime * Eq2_prime * (B1[0][1] * math.sin(delta1 - delta2) + G1[0][1] * math.cos(delta1 - delta2)) +
          Eq1_prime * Eq3_prime * (B1[0][2] * math.sin(delta1 - delta3) + G1[0][2] * math.cos(delta1 - delta3)))) / M[0]
    c = (abs(Ef[0]) - Eq1_prime + Id[0]*(xd[0]-xd_prime[0]))/Tdo_prime[0]
    d = omega2
    e = (Pm[1] - D[1]*omega2 - (Eq2_prime**2)*G1[1][1] -
         (Eq2_prime * Eq1_prime * (B1[1][0] * math.sin(delta2 - delta1) + G1[1][0] * math.cos(delta2 - delta1)) +
         Eq2_prime * Eq3_prime * (B1[1][2] * math.sin(delta2 - delta3) + G1[1][2] * math.cos(delta2 - delta3)))) / M[1]
    f = (abs(Ef[1]) - Eq2_prime + Id[1]*(xd[1]-xd_prime[1]))/Tdo_prime[1]
    g = omega3
    h = (Pm[2] - D[2]*omega3 - (Eq3_prime**2)*G1[2][2] -
         (Eq3_prime * Eq1_prime * (B1[2][1] * math.sin(delta3 - delta1) + G1[2][1] * math.cos(delta3 - delta1)) +
          Eq3_prime * Eq2_prime * (B1[2][0] * math.sin(delta3 - delta2) + G1[2][0] * math.cos(delta3 - delta2)))) / M[2]
    i = (abs(Ef[2]) - Eq3_prime + Id[2] * (xd[2] - xd_prime[2])) / Tdo_prime[2]


    return a, b, c, d, e, f, g, h, i



ic_fault_3rd_order = [delta0[0], 0, abs(E[0]), delta0[1], 0, abs(E[1]),  delta0[2], 0, abs(E[2])]


sol_fault_3rd_order = odeint(fault_3rd_order, ic_fault_3rd_order, t_fault)

delta1_fault_3rd_order = sol_fault_3rd_order[:, 0]
omega1_fault_3rd_order = sol_fault_3rd_order[:, 1]
Eq1_fault_3rd_order = sol_fault_3rd_order[:, 2]
delta2_fault_3rd_order = sol_fault_3rd_order[:, 3]
omega2_fault_3rd_order = sol_fault_3rd_order[:, 4]
Eq2_fault_3rd_order = sol_fault_3rd_order[:, 5]
delta3_fault_3rd_order = sol_fault_3rd_order[:, 6]
omega3_fault_3rd_order = sol_fault_3rd_order[:, 7]
Eq3_fault_3rd_order = sol_fault_3rd_order[:, 8]


E_post = [sol_fault_3rd_order[999, 2], sol_fault_3rd_order[999, 5], sol_fault_3rd_order[999, 8]]


def postfault_3rd_order(x, t):
    # Assign each ODE to vector element
    delta1 = x[0]
    omega1 = x[1]
    Eq1_prime = x[2]
    delta2 = x[3]
    omega2 = x[4]
    Eq2_prime = x[5]
    delta3 = x[6]
    omega3 = x[7]
    Eq3_prime = x[8]

    G = Y_post_reduced.real
    B = Y_post_reduced.imag

    # Define each ODE

    a = omega1
    b = (Pm[0] - D[0]*omega1 - (E_post[0]**2)*G[0][0] -
         (E_post[0] * E_post[1] * (B[0][1] * math.sin(delta1 - delta2) + G[0][1] * math.cos(delta1 - delta2)) +
          E_post[0] * E_post[2] * (B[0][2] * math.sin(delta1 - delta3) + G[0][2] * math.cos(delta1 - delta3)))) / M[0]
    c = (abs(Ef[0]) - Eq1_prime + Id[0] * (xd[0] - xd_prime[0])) / Tdo_prime[0]
    d = omega2
    e = (Pm[1] - D[1]*omega2 - (E_post[1]**2)*G[1][1] -
         (E_post[1] * E_post[0] * (B[1][0] * math.sin(delta2 - delta1) + G[1][0] * math.cos(delta2 - delta1)) +
          E_post[1] * E_post[2] * (B[1][2] * math.sin(delta2 - delta3) + G[1][2] * math.cos(delta2 - delta3)))) / M[1]
    f = (abs(Ef[1]) - Eq2_prime + Id[1] * (xd[1] - xd_prime[1])) / Tdo_prime[1]
    g = omega3
    h = (Pm[2] - D[2]*omega3 - (E_post[2]**2)*G[2][2] -
         (E_post[2] * E_post[0] * (B[2][0] * math.sin(delta3 - delta1) + G[2][0] * math.cos(delta3 - delta1)) +
          E_post[2] * E_post[1] * (B[2][1] * math.sin(delta3 - delta2) + G[2][1] * math.cos(delta3 - delta2)))) / M[2]
    i = (abs(Ef[2]) - Eq3_prime + Id[2] * (xd[2] - xd_prime[2])) / Tdo_prime[2]


    return a, b, c, d, e, f, g, h, i




ic_post_3rd_order = [sol_fault_3rd_order[999, 0], sol_fault_3rd_order[999, 1], sol_fault_3rd_order[999, 2],
          sol_fault_3rd_order[999, 3], sol_fault_3rd_order[999, 4], sol_fault_3rd_order[999, 5],
          sol_fault_3rd_order[999, 6], sol_fault_3rd_order[999, 7], sol_fault_3rd_order[999, 8]]

sol_post_3rd_order = odeint(postfault_3rd_order, ic_post_3rd_order, t_post)

delta1_post_3rd_order = sol_post_3rd_order[:, 0]
omega1_post_3rd_order = sol_post_3rd_order[:, 1]
Eq1_post_3rd_order = sol_post_3rd_order[:, 2]
delta2_post_3rd_order = sol_post_3rd_order[:, 3]
omega2_post_3rd_order = sol_post_3rd_order[:, 4]
Eq2_post_3rd_order = sol_post_3rd_order[:, 5]
delta3_post_3rd_order = sol_post_3rd_order[:, 6]
omega3_post_3rd_order = sol_post_3rd_order[:, 7]
Eq3_post_3rd_order = sol_post_3rd_order[:, 8]

# plot the results
# Plot of the angles
plt.plot(t_fault, deg(delta1_fault_3rd_order), color='blue', label=r'$\delta_{1}$')
plt.plot(t_fault, deg(delta2_fault_3rd_order), color='green', label=r'$\delta_{2}$')
plt.plot(t_fault, deg(delta3_fault_3rd_order), color='red', label=r'$\delta_{3}$')
plt.plot(t_post, deg(delta1_post_3rd_order), color='blue')
plt.plot(t_post, deg(delta2_post_3rd_order), color='green')
plt.plot(t_post, deg(delta3_post_3rd_order), color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor angles in degrees')
plt.title("Rotor angles using the third-order model")
plt.show()

# Plot of delta21 and delta 31
plt.plot(t_fault, deg(delta2_fault_3rd_order - delta1_fault_3rd_order), color='green', label=r'$\delta_{21}$')
plt.plot(t_fault, deg(delta3_fault_3rd_order - delta1_fault_3rd_order), color='red', label=r'$\delta_{31}$')
plt.plot(t_post, deg(delta2_post_3rd_order - delta1_post_3rd_order), color='green')
plt.plot(t_post, deg(delta3_post_3rd_order - delta1_post_3rd_order), color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor angles in degrees')
plt.title("Rotor angles with slack bus as reference using the third-order model")
plt.show()


# Plot of omegas
plt.plot(t_fault, omega1_fault_3rd_order/377+1, color='blue', label=r'$\omega_{1}$')
plt.plot(t_fault, omega2_fault_3rd_order/377+1, color='green', label=r'$\omega_{2}$')
plt.plot(t_fault, omega3_fault_3rd_order/377+1, color='red', label=r'$\omega_{3}$')
plt.plot(t_post, omega1_post_3rd_order/377+1, color='blue')
plt.plot(t_post, omega2_post_3rd_order/377+1, color='green')
plt.plot(t_post, omega3_post_3rd_order/377+1, color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor speed in radians per sec/377')
plt.title("Rotor speeds using the third-order model")
plt.show()


# Plot Eq
plt.plot(t_fault, Eq1_fault_3rd_order, color='blue', label=r'$E_{q1}$')
plt.plot(t_fault, Eq2_fault_3rd_order, color='green', label=r'$E_{q2}$')
plt.plot(t_fault, Eq3_fault_3rd_order, color='red', label=r'$E_{q3}$')
plt.plot(t_post, Eq1_post_3rd_order, color='blue')
plt.plot(t_post, Eq2_post_3rd_order, color='green')
plt.plot(t_post, Eq3_post_3rd_order, color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('E (p.u.)')
plt.title("Transient emfs using the third-order model")
plt.show()















import pandas as pd
import numpy as np
import math
import openpyxl
import cmath
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from PowerFlow_Project import V, betas, P, Q, Ybus, nbus, Y, Sb  # Should be Pk and Qk instead of P and Q


def rectangular(module, angle):
    return cmath.rect(module, (angle * math.pi) / 180)


def polar(number):
    return cmath.polar(number)[0], cmath.polar(number)[1] * (180 / math.pi)


def deg(angle):
    return (angle * 180) / math.pi


load_multiplier = [1, 1]  # new load = multiplier * previous load (how changes in load affect the dynamics)

# results calculated from power flow
print("V =", V)
print("theta =", betas * 180 / math.pi)
print("P =", P)
print("Q =", Q)

xd_prime = [0.08, 0.18, 0.12]
y14 = y41 = 1 / (xd_prime[0] * 1j)
y25 = y52 = 1 / (xd_prime[1] * 1j)
y36 = y63 = 1 / (xd_prime[2] * 1j)

ng = 3  # number of generators
nl = 2  # number of loads

# Step 2: admittances corresponding to the loads:
# y = S.conj/abs(V)**2

y77 = -(P[7 - ng - 1] * load_multiplier[0] / V[7 - ng - 1] ** 2 - Q[7 - ng - 1] * load_multiplier[0] * 1j / V[
    7 - ng - 1] ** 2)
y88 = -(P[8 - ng - 1] * load_multiplier[1] / V[8 - ng - 1] ** 2 - Q[8 - ng - 1] * load_multiplier[1] * 1j / V[
    8 - ng - 1] ** 2)

print(y77)
print(y88)

# Step 3: The internal generator voltages are calculated
E = np.zeros(ng, dtype=complex)
E_0 = np.zeros(ng, dtype=complex)
delta0 = np.zeros(ng)
for i in range(0, ng):
    E[i] = V[i] + Q[i] * xd_prime[i] / V[i] + (P[i] * xd_prime[i] / V[i]) * 1j
    print("E_", i + 1, "=", polar(E[i]))
    delta0[i] = np.angle(E[i]) + betas[i]
    print("delta_", i + 1, "=", deg(delta0[i]))
    E_0[i] = cmath.rect(abs(E[i]), delta0[i])  # use deg just for printing
    print("E_delta0 =", i + 1, "=", polar(E_0[i]))

# Step 4: Prefault, Faulted, and Postfault admittance matrices

# PREFAULT MATRIX

Y_pre = np.zeros((nbus + ng, nbus + ng), dtype=complex)

# Off diagonal Elements

# Elements that remain equal to the previous matrix
for i in range(0, nbus):
    for j in range(0, nbus):
        if i != j:
            Y_pre[i + ng][j + ng] = Ybus[i][j]

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
Y_fault = np.zeros((nbus + ng, nbus + ng), dtype=complex)

for i in range(nbus + ng):
    for j in range(nbus + ng):
        Y_fault[i][j] = Y_pre[i][j]

for i in range(nbus + ng):
    for j in range(nbus + ng):
        if i == bus_fault - 1:
            Y_fault[i][j] = 0
        if j == bus_fault - 1:
            Y_fault[i][j] = 0

print("Y faulted = ", Y_fault)

# POST FAULT MATRIX
# Line between buses 7-6 is removed

Y_post = np.zeros((nbus + ng, nbus + ng), dtype=complex)

for i in range(nbus + ng):
    for j in range(nbus + ng):
        Y_post[i][j] = Y_pre[i][j]

Y_post[5][5] = Y_pre[5][5] - Y[5]
Y_post[6][6] = Y_pre[6][6] - Y[5]
Y_post[5][6] = Y_pre[5][6] + Y[5]
Y_post[6][5] = Y_pre[6][5] + Y[5]

print(" Y post =", Y_post)

print(Y_pre[6][6])

# Kron reduction


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


# Kron with matrix products

# Y prefault matrix

Ynn1 = Y_pre[:3, :3]
Yns1 = Y_pre[:3, 3:]
Ysn1 = Y_pre[3:, :3]
Yss1 = Y_pre[3:, 3:]

Y_pre_reduced = Ynn1 - np.matmul(Yns1, (np.matmul(np.linalg.inv(Yss1), Ysn1)))

print(" Y pre reduced =", Y_pre_reduced)

# Y faulted matrix
# delete the row and column with zeros

Y_fault = np.delete(Y_fault, 6, 0)
Y_fault = np.delete(Y_fault, 6, 1)

Ynn2 = Y_fault[:3, :3]
Yns2 = Y_fault[:3, 3:]
Ysn2 = Y_fault[3:, :3]
Yss2 = Y_fault[3:, 3:]

Y_fault_reduced = Ynn2 - np.matmul(Yns2, (np.matmul(np.linalg.inv(Yss2), Ysn2)))

print(" Y fault reduced =", Y_fault_reduced)

# Y post-fault matrix

Ynn3 = Y_post[:3, :3]
Yns3 = Y_post[:3, 3:]
Ysn3 = Y_post[3:, :3]
Yss3 = Y_post[3:, 3:]

Y_post_reduced = Ynn3 - np.matmul(Yns3, (np.matmul(np.linalg.inv(Yss3), Ysn3)))

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
H = np.array([10, 3.01, 6.4])
f0 = 60
M = H / (math.pi * f0)


# prefault numerical integration


def fault(x, t):
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
    b = (Pm[0] - (abs(E[0]) ** 2) * G1[0][0] -
         (abs(E[0]) * abs(E[1]) * (B1[0][1] * math.sin(delta1 - delta2) + G1[0][1] * math.cos(delta1 - delta2)) +
          abs(E[0]) * abs(E[2]) * (B1[0][2] * math.sin(delta1 - delta3) + G1[0][2] * math.cos(delta1 - delta3)))) / M[0]
    c = omega2
    d = (Pm[1] - (abs(E[1]) ** 2) * G1[1][1] -
         (abs(E[1]) * abs(E[0]) * (B1[1][0] * math.sin(delta2 - delta1) + G1[1][0] * math.cos(delta2 - delta1)) +
          abs(E[1]) * abs(E[2]) * (B1[1][2] * math.sin(delta2 - delta3) + G1[1][2] * math.cos(delta2 - delta3)))) / M[1]
    e = omega3
    f = (Pm[2] - (abs(E[2]) ** 2) * G1[2][2] -
         (abs(E[2]) * abs(E[0]) * (B1[2][0] * math.sin(delta3 - delta1) + G1[2][0] * math.cos(delta3 - delta1)) +
          abs(E[2]) * abs(E[1]) * (B1[2][1] * math.sin(delta3 - delta2) + G1[2][1] * math.cos(delta3 - delta2)))) / M[2]

    return a, b, c, d, e, f


t_fault = np.linspace(0, 0.10, 1000)
ic_fault = [delta0[0], 0, delta0[1], 0, delta0[2], 0]

sol_fault = odeint(fault, ic_fault, t_fault)

delta1_fault = sol_fault[:, 0]
omega1_fault = sol_fault[:, 1]
delta2_fault = sol_fault[:, 2]
omega2_fault = sol_fault[:, 3]
delta3_fault = sol_fault[:, 4]
omega3_fault = sol_fault[:, 5]


def postfault(x, t):
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
    b = (Pm[0] - (abs(E[0]) ** 2) * G[0][0] -
         (abs(E[0]) * abs(E[1]) * (B[0][1] * math.sin(delta1 - delta2) + G[0][1] * math.cos(delta1 - delta2)) +
          abs(E[0]) * abs(E[2]) * (B[0][2] * math.sin(delta1 - delta3) + G[0][2] * math.cos(delta1 - delta3)))) / M[0]
    c = omega2
    d = (Pm[1] - (abs(E[1]) ** 2) * G[1][1] -
         (abs(E[1]) * abs(E[0]) * (B[1][0] * math.sin(delta2 - delta1) + G[1][0] * math.cos(delta2 - delta1)) +
          abs(E[1]) * abs(E[2]) * (B[1][2] * math.sin(delta2 - delta3) + G[1][2] * math.cos(delta2 - delta3)))) / M[1]
    e = omega3
    f = (Pm[2] - (abs(E[2]) ** 2) * G[2][2] -
         (abs(E[2]) * abs(E[0]) * (B[2][0] * math.sin(delta3 - delta1) + G[2][0] * math.cos(delta3 - delta1)) +
          abs(E[2]) * abs(E[1]) * (B[2][1] * math.sin(delta3 - delta2) + G[2][1] * math.cos(delta3 - delta2)))) / M[2]

    return a, b, c, d, e, f


t_post = np.linspace(0.1, 1.5, 1000)

ic_post = [sol_fault[999, 0], sol_fault[999, 1], sol_fault[999, 2],
           sol_fault[999, 3], sol_fault[999, 4], sol_fault[999, 5]]

sol_post = odeint(postfault, ic_post, t_post)

delta1_post = sol_post[:, 0]
omega1_post = sol_post[:, 1]
delta2_post = sol_post[:, 2]
omega2_post = sol_post[:, 3]
delta3_post = sol_post[:, 4]
omega3_post = sol_post[:, 5]

# plot the results
# Plot of the angles
plt.plot(t_fault, deg(delta1_fault), color='blue', label=r'$\delta_{1}$')
plt.plot(t_fault, deg(delta2_fault), color='green', label=r'$\delta_{2}$')
plt.plot(t_fault, deg(delta3_fault), color='red', label=r'$\delta_{3}$')
plt.plot(t_post, deg(delta1_post), color='blue')
plt.plot(t_post, deg(delta2_post), color='green')
plt.plot(t_post, deg(delta3_post), color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor angles in degrees')
plt.show()

# Plot of delta21 and delta 31
plt.plot(t_fault, deg(delta2_fault - delta1_fault), color='green', label=r'$\delta_{21}$')
plt.plot(t_fault, deg(delta3_fault - delta1_fault), color='red', label=r'$\delta_{31}$')
plt.plot(t_post, deg(delta2_post - delta1_post), color='green')
plt.plot(t_post, deg(delta3_post - delta1_post), color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor angles in degrees')
plt.show()

# Plot of omegas
plt.plot(t_fault, omega1_fault / 377 + 1, color='blue', label=r'$\omega_{1}$')
plt.plot(t_fault, omega2_fault / 377 + 1, color='green', label=r'$\omega_{2}$')
plt.plot(t_fault, omega3_fault / 377 + 1, color='red', label=r'$\omega_{3}$')
plt.plot(t_post, omega1_post / 377 + 1, color='blue')
plt.plot(t_post, omega2_post / 377 + 1, color='green')
plt.plot(t_post, omega3_post / 377 + 1, color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor speed in radians per sec/377')
plt.show()

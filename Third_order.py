# Third Order Model
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Second_order import Y_fault_reduced, Y_post_reduced, Pm, E, delta0, ng, M, betas, xd_prime
from PowerFlow_Project import V, Q, P


def deg(angle):
    return (angle * 180) / math.pi
def polar(number):
    return (cmath.polar(number)[0], cmath.polar(number)[1] * (180 / math.pi))

# Define variables for third order model according to the book


Tdo_prime = [0.84, 0.83, 0.81] # transient d-axis time constants
xd = [1.8, 1.7, 1.5]  # d-axis steady-state reactances
time_fault = 0.1   # Fault clearing time
time_simulation = 5  # Simulation time
D = [0.1, 0.08, 50]    # Damping for each generator


#xd = [0.57392, 1.29132, 0.86088]
#xd = [0.5, 0.5, 0.1]

# Load increase





# Finding Ef
# Ed0 = 0 and E'd0 = 0 as in salient pole generators

Ef = np.zeros(ng, dtype=complex)
Ef_0 = np.zeros(ng, dtype=complex)
delta_Ef = np.zeros(ng)



for i in range(0, ng):
    Ef[i] = V[i]+Q[i]*xd[i]/V[i] + (P[i]*xd[i]/V[i])*1j
    delta_Ef[i] = np.angle(Ef[i]) + betas[i]
    Ef_0[i] = cmath.rect(abs(Ef[i]), delta_Ef[i])    #use deg just for printing
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



def fault(x, t):
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



t_fault = np.linspace(0, time_fault, 1000)
ic_fault = [delta0[0], 0, abs(E[0]), delta0[1], 0, abs(E[1]),  delta0[2], 0, abs(E[2])]


sol_fault = odeint(fault, ic_fault, t_fault)

delta1_fault = sol_fault[:, 0]
omega1_fault = sol_fault[:, 1]
Eq1_fault = sol_fault[:, 2]
delta2_fault = sol_fault[:, 3]
omega2_fault = sol_fault[:, 4]
Eq2_fault = sol_fault[:, 5]
delta3_fault = sol_fault[:, 6]
omega3_fault = sol_fault[:, 7]
Eq3_fault = sol_fault[:, 8]


E_post = [sol_fault[999, 2], sol_fault[999, 5], sol_fault[999, 8]]
# E_post = [1.11, 1.06, 1.18]

def postfault(x, t):
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


t_post = np.linspace(time_fault, time_simulation, 1000)

ic_post = [sol_fault[999, 0], sol_fault[999, 1], sol_fault[999, 2],
          sol_fault[999, 3], sol_fault[999, 4], sol_fault[999, 5],
          sol_fault[999, 6], sol_fault[999, 7], sol_fault[999, 8]]

sol_post = odeint(postfault, ic_post, t_post)

delta1_post = sol_post[:, 0]
omega1_post = sol_post[:, 1]
Eq1_post = sol_post[:, 2]
delta2_post = sol_post[:, 3]
omega2_post = sol_post[:, 4]
Eq2_post = sol_post[:, 5]
delta3_post = sol_post[:, 6]
omega3_post = sol_post[:, 7]
Eq3_post = sol_post[:, 8]

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
plt.plot(t_fault, omega1_fault/377+1, color='blue', label=r'$\omega_{1}$')
plt.plot(t_fault, omega2_fault/377+1, color='green', label=r'$\omega_{2}$')
plt.plot(t_fault, omega3_fault/377+1, color='red', label=r'$\omega_{3}$')
plt.plot(t_post, omega1_post/377+1, color='blue')
plt.plot(t_post, omega2_post/377+1, color='green')
plt.plot(t_post, omega3_post/377+1, color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('Rotor speed in radians per sec/377')
plt.show()


# Plot Eq
plt.plot(t_fault, Eq1_fault, color='blue', label=r'$E_{q1}$')
plt.plot(t_fault, Eq2_fault, color='green', label=r'$E_{q2}$')
plt.plot(t_fault, Eq3_fault, color='red', label=r'$E_{q3}$')
plt.plot(t_post, Eq1_post, color='blue')
plt.plot(t_post, Eq2_post, color='green')
plt.plot(t_post, Eq3_post, color='red')
plt.legend()
plt.grid()
plt.xlabel('Time in seconds')
plt.ylabel('E (p.u.)')
plt.show()


print("E post =", E_post)
print("ic Post =", ic_post)
print("delta0 =", deg(delta0))



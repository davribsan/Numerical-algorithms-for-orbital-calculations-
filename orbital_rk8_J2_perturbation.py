'''
(a) Orbital Mechanics Solver with J2 based on Runge-Kutta 8th order 
(b) Version number: 1
(c) Autors: davribsan
(d) Date of initializaition: 13/10/25
(e) Description of the program: 
    Given an orbit, compute the exact solution of the central-body problem, and integrate the 
    equations of motion of a satellite around a point-like Earth to study numerical convergence 
    using a Runge-Kutta 8th order numerical solver with J2 perturbations.

(f) Numerical solver: Runnge-Kutta 8th order
    References:
    SciPy Developers. (2025). scipy.integrate.DOP853. 
    Url: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html
    
g) Range of validity expected of the parameters and range tested:
    - 10 orbits with e = 0.9 and a = 70000km, J2 and an integration step size of 1 s.
   
(h) Inputs:

    G_EARTH: Earth gravitational constant (constant)                                        [m3⋅kg-1⋅s-2]
    M_EARTH: Mass of Earth (constant)                                                       [kg]
    R_EARTH: Earth radius (constant)                                                        [m]
    J2: J2 orbital perturbation (constant)                                                  [ ]
    mass_sat: Mass of the satellite                                                         [kg] 
    a_B: Semimajor axis of the orbit from the problem assignament                           [m]
    e_B: Eccentricity of the orbit from the probelm assignament                             [ ]
    i_B: Inclination of the orbit from the problem assignement                              [º]
    w0: Initial argument of the periapsis                                                   [º]
    omega0: Initial right ascension of the ascending node                                   [º]
    initial_true_anomaly: Initial true anomaly                                              [º]
    n_orbits: Number of orbits to compute                                                   [ ]
    delta_t_values: List with integration step times                                        [s]

(i) Outputs:

    Orbital period                                                                          [s]
    Instant of time                                                                         [s]
    Position of the spacecraft at any instant                                               [m]
    Velocity of spacecraft at any instant                                                   [m/s]
    Time of computantion of the full trajectory                                             [s]
    Numerical argument of the perigee shift rate                                            [º/s]
    Analytical argument of the perigee shift rate                                           [º/s]
    Numerical argument of the perigee shift per orbit                                       [º]
    Analytical argument of the perigee shift per orbit                                      [º]
    Difference between analytical and numerical argument of the perigee shift rate          [º/s]
    Difference between analytical and numerical argument of the perigee shift per orbit     [º]
    Numerical RAAN shift rate                                                               [º/s]
    Analytical RAAN shift rate                                                              [º/s]
    Numerical RAAN shift per orbit                                                          [º]
    Analytical RAAN shift per orbit                                                         [º]
    Difference between analytical and numerical RAAN shift rate                             [º/s]
    Difference between analytical and numerical RAAN shift per orbit                        [º]
    Orbital elements graphs
    Numerical trajectory graph
    
(j) List of dependencies:
    - This program requires solve_ivp numerical integrator with the Dormand-Prince method ('DOP853') 
    - This program requires the script "utils"
 
(k) Software version: Python 3.12.4
'''

# ------------------------------------------------------------------------------------
# Libraries 

import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from auxiliar import utils

# ------------------------------------------------------------------------------------
# Constants

G_EARTH = 6.6743e-11 
M_EARTH = 5.97219e24 
R_EARTH = 6378e3
J2 = 1.08262668e-3

# ------------------------------------------------------------------------------------
# Functions 

def two_body(t, w):
    "Derivative of the state vector w for a two-body problem"
    #w = [x_e, y_e, z_e, vx_e, vy_e, vz_e, x_s, y_s, z_s, vx_s, vy_s, vz_s]
    #m = [M_EARTH, mass_sat]
    
    mu = G_EARTH * M_EARTH
    
    # Earth position and velocity
    p_E = w[0:3]
    v_E = w[3:6]
    
    # Satellite position and velocity
    p_s = w[6:9]
    v_s = w[9:12]
    
    # Satellite acceleration vector
    r_vec = p_s - p_E
    x,y,z = r_vec
    r = np.linalg.norm(r_vec)

    # Central body acceleration
    a_central = -mu * r_vec / r**3

    # J2 perturbation
    common_factor = (3*J2*mu*(R_EARTH**2)) / (2*(r**5))    
    z2_r2 = (z**2) / (r**2)

    a_J2_x = common_factor * x * (5 * z2_r2 - 1)
    a_J2_y = common_factor * y * (5 * z2_r2 - 1)
    a_J2_z = common_factor * z * (5 * z2_r2 - 3)
    a_J2 = np.array([a_J2_x, a_J2_y, a_J2_z])
    
    # Total acceleration
    a_s = a_central + a_J2

    # Acceleration of the Earth
    a_E = np.zeros(3) # Considering fixed
    
    # Derivatives, dx/dt = v, dv/dt = a
    derivatives = np.concatenate([v_E, a_E, v_s, a_s])
    return derivatives

def Rz(angle):
    "Computes the rotation matrix in z axis"
    angle = math.radians(angle)
    c = math.cos(angle) 
    s = math.sin(angle)
    return np.array([[c, -s, 0.0],[s, c, 0.0],[0.0, 0.0, 1.0]])

def Rx(angle):
    "Computes the rotation matrix in x axis"
    angle = math.radians(angle)
    c = math.cos(angle) 
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0],[0.0, c, -s],[0.0, s, c]])

def argument_periapsis_analytical_variation(mu,a,e,i):
    "Computes the analytical mean variation of the argument of the periapsis in º/s"
    # Mean motion
    n = utils.mean_motion(mu,a)

    # Mean variation of the argument of the periapsis
    i = math.radians(i)
    w_shift_ratio = ((3*n*J2*(R_EARTH**2))/(2*(a**2)*((1-e**2)**2)))*(2-2.5*(math.sin(i))**2)
    return math.degrees(w_shift_ratio)

def raan_analytical_variation(mu,a,e,i):
    "Computes the analytical mean variation of the argument of the periapsis in º/s"
    # Mean motion
    n = utils.mean_motion(mu,a)

    # Mean variation of the argument of the periapsis
    i = math.radians(i)
    omega_shift_ratio = -((3*n*J2*(R_EARTH**2))/(2*(a**2)*((1-e**2)**2)))*math.cos(i)
    return math.degrees(omega_shift_ratio)

# ------------------------------------------------------------------------------------
# Inputs 

mass_sat = 700 
a_B = 70000e3              
e_B = 0.9
i_B = 30       
w0 = 0         
omega0 = 0    
initial_true_anomaly = 0  
n_orbits = 10 
delta_t_values = [1]

# ------------------------------------------------------------------------------------
# Program

# Save all results printed on the screen
sys.stdout = open("RK8_results_J2.txt", "w")

# Standard gravitational parameter 
mu_earth = G_EARTH*M_EARTH 


# Numerical trajectory
# 1. Radius of the true anomaly initial/final truen anomalies
r0 = utils.orbit_radius_from_anomaly(a_B, e_B, initial_true_anomaly)

# 2. Coordinates of initial true anomaly at 2D plane 
p0 = utils.compute_position(r0,initial_true_anomaly) 

# 3. Velocity at initial true anomaly at 2D plane
v0 = utils.velocity_components(a_B,e_B,mu_earth,initial_true_anomaly,r0)

# 4. Convert p0 and v0 from 2D to 3D (add inclination)
R = Rz(omega0) @ Rx(i_B) @ Rz(w0)
p0_eci = R @ p0
v0_eci = R @ v0
x0, y0, z0 = p0_eci
vx_0, vy_0, vz_0 = v0_eci

# 5. Initial estates
earth = [0, 0, 0, 0, 0, 0] 
sat = [x0, y0, z0, vx_0, vy_0, vz_0]

# 6. Join estates (because of the R-K 8 differential equation)
initial_state = np.array(earth + sat)
m = np.array([M_EARTH, mass_sat]) 

# 7. Time of flight
T = utils.orbital_period(a_B,mu_earth)
print(f'\nOrbital period: {T/(3600)} h')
tof = T*n_orbits

# 7.1. Interval of integration
t_span = (0, tof)

# 7.2. Times at which to store the computed solution
t_eval = np.arange(0, tof, delta_t_values[0])

# 8. Initialize timer
start_time = time.time()

# 9. Numerical trajectory
sol = solve_ivp(two_body, t_span, initial_state, method='DOP853', t_eval=t_eval, rtol=1e-9, atol=1e-12)

# 10. Stop timer
stop_time = time.time()
print(f"Time of computation for {n_orbits} orbits with a integration interval of {delta_t_values[0]} s: {stop_time-start_time} s\n")

# 11. Satellite numerical position
x_solver = sol.y[6, :].T
y_solver = sol.y[7, :].T
z_solver = sol.y[8, :].T
positions_num = sol.y[6:9, :].T  # satellite x, y, z

# 12. Satellite numerical velocity
velocities_num = sol.y[9:12, :].T # satellite vx, vy, vz
v_magnitudes = np.linalg.norm(velocities_num, axis=1)

# 13. Time at which each position is computed 
times = sol.t

# 14. Radius at each point of the numerical trajectory
r_solver = utils.orbit_radii_from_coordinates(positions_num)

# 15. Perigee position
perigee_indices_num = argrelextrema(r_solver, np.less)[0]  
perigee_positions_num = positions_num[perigee_indices_num]


#Classical orbit elements (effect of J2 perturbation)
# 1. Radii at perigee
radii_at_perigee =  utils.orbit_radii_from_coordinates(perigee_positions_num)

a_num = []
e_num = []
i_num = []
omega_num = []
w_num = []

for i, p_index in enumerate(perigee_indices_num):
    r = radii_at_perigee[i]
    v = v_magnitudes[p_index]
    p = perigee_positions_num[i]

    # 2. Semimajor axis
    a_num.append(utils.compute_semimajor_axis(r,v, mu_earth))

    # 3. Eccentricity
    e_num.append(utils.compute_eccentricity(p,velocities_num[p_index,:],mu_earth))

    # 4. Inclination
    i_num.append(utils.inclination(p,velocities_num[p_index,:]))

    # 5. Right ascen¡sion of the acessing node
    omega_num.append(utils.raan(p,velocities_num[p_index,:]))

    # 6. Argument of the perigee
    w_num.append(utils.argumet_periapsis(mu_earth,p,velocities_num[p_index,:]))

# 7. Apogee
apogee_indices_num  = argrelextrema(r_solver, np.greater)[0] 
apogee_positions_num = positions_num[apogee_indices_num]
radii_at_apogee =  utils.orbit_radii_from_coordinates(apogee_positions_num)


# Quantify the shift
# 1. Argument of the perigee
w_num = np.array(w_num)  
w_shift_num = np.diff(w_num) #Change between orbits 
w_shift_num = np.mod(w_shift_num, 360)     
w_shift_num = np.where(w_shift_num > 180, 360 - w_shift_num, w_shift_num) 
mean_w_shift_orbit = np.mean(w_shift_num)
numerical_w_shift_rate = mean_w_shift_orbit/T
print(f'Numerical argument of the perigee shift rate computed from {n_orbits} orbits with a integration interval of {delta_t_values[0]} s: {numerical_w_shift_rate} º/s')

# 1.2. Compare with the analytical J2 approximation
# 1.2.1. Compute the rate
analytical_w_shift_rate = argument_periapsis_analytical_variation(mu_earth,a_B,e_B,i_B)
print(f'Analytical argument of the perigee shift rate: {analytical_w_shift_rate} º/s')

# 1.2.2. Computes the variation per orbit
w_shift_exact_orbit = analytical_w_shift_rate*T
print(f'Numerical argument of the perigee shift per orbit computed from {n_orbits} orbits with a integration interval of {delta_t_values[0]} s: {mean_w_shift_orbit} º')
print(f'Analytical argument of the perigee shift per orbit: {w_shift_exact_orbit} º')

# 1.2.3. Comparison
print(f'Difference between analytical and numerical argument of the perigee shift rate: {np.abs(analytical_w_shift_rate-numerical_w_shift_rate)} º/s')
print(f'Difference between analytical and numerical argument of the perigee shift per orbit: {np.abs(w_shift_exact_orbit-mean_w_shift_orbit)} º \n')

# 2. Argument of the perigee
omega_num = np.array(omega_num)  
omega_shift_num = np.diff(omega_num) #Change between orbits 
omega_shift_num = np.mod(omega_shift_num, 360)     
omega_shift_num = np.where(omega_shift_num > 180, 360 - omega_shift_num, omega_shift_num) 
mean_omega_shift_orbit = -np.mean(omega_shift_num)
numerical_omega_shift_rate = mean_omega_shift_orbit/T
print(f'Numerical RAAN shift rate computed from {n_orbits} orbits with a integration interval of {delta_t_values[0]} s: {numerical_omega_shift_rate} º/s')

# 2.1. Compare with the analytical J2 approximation
# 2.1.1. Compute the rate
analytical_omega_shift_rate = raan_analytical_variation(mu_earth,a_B,e_B,i_B)
print(f'Analytical RAAN shift rate: {analytical_omega_shift_rate} º/s')

# 2.1.2. Computes the variation per orbit
omega_shift_exact_orbit = analytical_omega_shift_rate*T
print(f'Numerical RAAN shift per orbit computed from {n_orbits} orbits with a integration interval of {delta_t_values[0]} s: {mean_omega_shift_orbit} º')
print(f'Analytical RAAN shift per orbit: {omega_shift_exact_orbit} º')

# 2.2.3. Comparison
print(f'Difference between analytical and numerical RAAN shift rate: {np.abs(analytical_omega_shift_rate-numerical_omega_shift_rate)} º/s')
print(f'Difference between analytical and numerical RAAN shift per orbit: {np.abs(omega_shift_exact_orbit-mean_omega_shift_orbit)} º \n')


# Save results in a txt file
sys.stdout.close()
sys.stdout = sys.__stdout__


# Plot results
# 1. Inclination 
orbit_numbers = np.arange(1,len(perigee_indices_num)+1)
orbit_numbers = orbit_numbers.astype(str).tolist()
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers,i_num, color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 Inclination change with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('i [º]')
plt.savefig('RK8_inclination_change_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 2. Right ascension of the ascending node 
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers,omega_num, color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 RAAN change with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('Ω [º]')
plt.savefig('RK8_raan_change_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 3. Argument of the perigee
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers,w_num, color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 Argument of the perigee change with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('w [º]')
plt.savefig('RK8_arg(perigee)_change_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 4 Semimajor axis
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers,a_num, color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 Semimajor axis with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('a [m]')
plt.savefig('RK8_semimajor_axis_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 5. Eccentricity
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers,e_num, color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 Eccentricity with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('e')
plt.savefig('RK8_eccentricity_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 6. Perigee
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers,radii_at_perigee, color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 Radii at perigee with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('Radius [m]')
plt.savefig('RK8_radii_perigee_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 7. Apogee
plt.figure(figsize=(10,6))
min_len = min(len(orbit_numbers), len(radii_at_apogee))
plt.scatter(orbit_numbers[:min_len],radii_at_apogee[:min_len], color = 'blue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'RK8 Radii at apogee with J2 perturbation', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('Radius [m]')
plt.savefig('RK8_radii_apogee_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

# 8 Plot trajectories 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = R_EARTH * np.outer(np.cos(u), np.sin(v))
y = R_EARTH * np.outer(np.sin(u), np.sin(v))
z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=1) 
ax.plot(positions_num[:,0], positions_num[:,1], positions_num[:,2], color='r',linewidth=0.2)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('RK8 orbits with J2 perturbation')

# Adjust limits for proportional representation
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
all_x = np.concatenate([x.flatten(), positions_num[:,0]])
all_y = np.concatenate([y.flatten(), positions_num[:,1]])
all_z = np.concatenate([z.flatten(), positions_num[:,2]])

max_range = np.array([all_x.max()-all_x.min(),
                      all_y.max()-all_y.min(),
                      all_z.max()-all_z.min()]).max() / 2

mid_x = (all_x.max() + all_x.min()) / 2
mid_y = (all_y.max() + all_y.min()) / 2
mid_z = (all_z.max() + all_z.min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.savefig('RK8_trajectory_J2.png', dpi=500,  bbox_inches='tight')
plt.show()

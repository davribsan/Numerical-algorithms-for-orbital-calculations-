'''
(a) Orbital Mechanics Solver based on Runge-Kutta 8th order 
(b) Version number: 1
(c) Autors: davribsan
(d) Date of initializaition: 13/10/25
(e) Description of the program: 
    Given an orbit, compute the exact solution of the central-body problem, and integrate the 
    equations of motion of a satellite around a point-like Earth to study numerical convergence 
    using a Runge-Kutta 8th order numerical solver. The problem is considered in only two dimensions.

(f) Numerical solver: Runnge-Kutta 8th order
    References:
    SciPy Developers. (2025). scipy.integrate.DOP853. 
    Url: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html
    
g) Range of validity expected of the parameters and range tested
   - Expected: 5 orbits with e = 0.9 and a = 70000km and an integration step size of 100 s.
   - Tested: 5 orbits with e = 0.9 and a = 70000km and an integration step size of [100, 50, 20, 10, 5] s.
             100 orbits with e = 0.9 and a = 70000km and an integration step size of 1 s.

(h) Inputs:

    G_EARTH: Earth gravitational constant (constant)                 [m3⋅kg-1⋅s-2]
    M_EARTH: Mass of Earth (constant)                                [kg]
    R_EARTH: Earth radius (constant)                                 [m]
    mass_sat: Mass of the satellite                                  [kg] 
    a_B: Semimajor axis of the orbit from the problem assignament    [m]
    e_B: Eccentricity of the orbit from the probelm assignament      [ ]
    initial_true_anomaly: Initial true anomaly                       [º]
    n_orbits: Number of orbits to compute                            [ ]
    delta_t_values: List with integration step times                 [s]

(i) Outputs:
    Numerical and analytical trajectories
    Graph of radius/true anomaly/position error for fixed step integration time
    Graph of position erros for different intgration step times
    Grapgh pf computational time for different integration step times
    Graph of perigee error
    Graph of apogee error
    
(j) List of dependencies:
    - This program requires solve_ivp numerical integrator with the Dormand-Prince method ('DOP853')  
    - This program requires the script "utils.py"
    
(k) Software version: Python 3.12.4
'''

# ------------------------------------------------------------------------------------
# Libraries 

import numpy as np
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

# ------------------------------------------------------------------------------------
# Functions 

def two_body(t, w):
    "Derivative of the state vector w for a two-body problem"

    mu = G_EARTH * M_EARTH
    
    # Earth position and velocity
    p_E = w[0:3]
    v_E = w[3:6]
    
    # Satellite position and velocity
    p_s = w[6:9]
    v_s = w[9:12]
    
    # Satellite acceleration vector
    r_vec = p_s - p_E
    r_norm = np.linalg.norm(r_vec)
    a_s = -mu * r_vec / r_norm**3
    
    # Acceleration of the Earth
    a_E = np.zeros(3) # Considering fixed
    
    # Derivatives, dx/dt = v, dv/dt = a
    derivatives = np.concatenate([v_E, a_E, v_s, a_s])
    return derivatives

# ------------------------------------------------------------------------------------
# Inputs 

mass_sat = 700 
a_B = 70000e3              
e_B = 0.9    
initial_true_anomaly = 0  
n_orbits = 5 
delta_t_values = [100, 50, 20, 10, 5]

# ------------------------------------------------------------------------------------
# Program

# Save all results printed on the screen
sys.stdout = open("RK8_results.txt", "w")

# Standard gravitational parameter 
mu_earth = G_EARTH*M_EARTH 


# Numerical trajectory
# 1. Radius of the true anomaly initial/final truen anomalies
r = utils.orbit_radius_from_anomaly(a_B, e_B, initial_true_anomaly)

# 2. Coordinates of initial/final true anomaly 
p = utils.compute_position(r,initial_true_anomaly) 
x,y,z = p

# 3. Velocity at initial/final true anomaly
v_theoretical = utils.orbit_velocity(mu_earth,a_B,r)

# 4. Velocity components at initial/final true anomaly
vx,vy,vz = utils.velocity_components(a_B,e_B,mu_earth,initial_true_anomaly,r)

# 5. Initial estates
earth = [0, 0, 0, 0, 0, 0] 
sat = [x, y, z, vx, vy, vz]

# 6. Join estates (because of the R-K 8 differential equation)
w0 = np.array(earth + sat)
m = np.array([M_EARTH, mass_sat]) 

# 7. Time of flight
T = utils.orbital_period(a_B,mu_earth)
print('\nOrbital period:',T/(3600))
tof = T*n_orbits

# 7.1. Interval of integration
t_span = (0, tof)

# 7.2. Times at which to store the computed solution
t_eval = np.arange(0, tof, delta_t_values[0])

# 8. Initialize timer
start_time = time.time()

# 9. Numerical trajectory
sol = solve_ivp(two_body, t_span, w0, method='DOP853', t_eval=t_eval, rtol=1e-9, atol=1e-12)

# 10. Stop timer
stop_time = time.time()
print(f"RK8 Time of computation for {n_orbits} orbits with an integration interval of {delta_t_values[0]} s: {stop_time-start_time} s")

# 11. Satellite numerical position
x_solver = sol.y[6, :].T
y_solver = sol.y[7, :].T
z_solver = sol.y[8, :].T
positions_num = sol.y[6:9, :].T  # satellite x, y, z

# 12. Satellite numerical velocity
velocities_num = sol.y[9:12, :].T # satellite vx, vy, vz

# 13. Radius at each point of the numerical trajectory
r_solver = utils.orbit_radii_from_coordinates(positions_num)

# 14. True anomalies at each point of the numerical trajectory
true_anomalies_solver = []
for i in range(len(r_solver)):

    r = r_solver[i]
    p = positions_num[i,:]
    v_ = velocities_num[i,:]
    true_anomalies_solver.append(utils.compute_true_anomaly_from_r(mu_earth,p,v_))

true_anomalies_solver = np.array(true_anomalies_solver)


# Analytical trajectory
# 1.Initial eccentric anomaly
E_0 = utils.eccentric_anomaly(e_B,initial_true_anomaly)

# 2. Initial mean anomaly 
M_0 = utils.mean_anomaly_from_eccentric_anomaly(E_0,e_B)

# 3. Mean motion
n = utils.mean_motion(mu_earth,a_B)

# 4. True anomaly at each instant of time 
true_anomalies_exact = []
t = 0 #Initial position
for _ in range(positions_num.shape[0]):

    # 4.1. Mean anomaly  
    M = utils.mean_anomaly_from_mean_motion(M_0,n,t)

    # 4.2. Eccentric anomaly from mean anomaly
    E = utils.solve_Kepler(e_B, M)

    # 4.3. True anomaly
    true_anomalies_exact.append(utils.compute_true_anomaly_from_E(e_B,E))

    # 4.4 Update time 
    t = t + delta_t_values[0]

true_anomalies_exact = np.array(true_anomalies_exact)

# 5. Radius at each point of the analytical trajectory 
r_exact = []
for theta in true_anomalies_exact:
    r_exact.append(utils.orbit_radius_from_anomaly(a_B,e_B,theta))

r_exact = np.array(r_exact)

# 6. Position at each true anomaly of the analytical trajectory
positions_exact = np.array([utils.compute_position(r, theta) for r, theta in zip(r_exact, true_anomalies_exact)])
x_exact = positions_exact[:,0]
y_exact = positions_exact[:,1]
z_exact = positions_exact[:,2]


# Plot trajectories
plt.figure(figsize=(6,5))

# 1. Analytical trajectory
plt.plot(x_exact,y_exact, color = 'g', linewidth = 2, label='Analytical trajectory')

# 2. Numerical trajectory
plt.plot(x_solver,y_solver, color = 'r', linewidth = 0.5, linestyle='--', label='Numerical trajectory')

# Plot the Earth
angle = np.linspace(0, 2*np.pi, 200)
x_circle = R_EARTH * np.cos(angle)
y_circle = R_EARTH * np.sin(angle)
plt.fill(x_circle, y_circle, color='b', alpha=1,     label='Earth')

plt.title('RK8 Numerical and Analytical trajectories', fontweight='bold', fontsize=12)
plt.xlabel('X[m]')
plt.ylabel('Y[m]')
plt.legend(loc = 'upper right')
plt.axis('equal')
plt.savefig('RK8_trajectories.png', dpi=500, bbox_inches='tight')
plt.show()


# Differences between analytical and numerical trajectories 
# 1. Radii
delta_r = np.abs(r_exact-r_solver)

# 2. True anomaly
delta_theta = np.abs(true_anomalies_exact-true_anomalies_solver)

# 3. Position error
pos_error = np.linalg.norm((positions_num-positions_exact),axis=1)

# 4. Representation
instant_time = np.arange(0,tof, delta_t_values[0]) 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9,6))

# 4.1. Numerical-analytical radii diference
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
ax1.scatter(instant_time/3600, delta_r, color='dodgerblue', s=1)
ax1.set_ylabel('Δr [m]')
ax1.set_title('RK8 radii difference between analytical and numerical trajectories', fontweight='bold', fontsize=14)

# 4.2. Numerical-analytical true anomaly diference 
max_anomaly_error = 20
n_outliers = np.sum(delta_theta > max_anomaly_error) # Discard outliers for representation purposes
delta_theta[delta_theta > max_anomaly_error] = 0
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
ax2.scatter(instant_time/3600, delta_theta, color='dodgerblue', s=1)
ax2.set_ylabel('Δθ [º]')
ax2.text(0.01, 0.8, f'Discarding {n_outliers} outliers > {max_anomaly_error}º for representation effects',
         fontsize=12, color='red', ha='left', va='bottom', transform=ax2.transAxes)
ax2.set_title('RK8 true anomaly difference between analytical and numerical trajectories', fontweight='bold', fontsize=14)

# 4.3. Numerical-analytical position diference  
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
ax3.scatter(instant_time/3600, pos_error, color='dodgerblue', s=1)
ax3.set_ylabel('Δp [m]')
ax3.set_xlabel('Time of flight [h]')
ax3.set_title('RK8 position error between analytical and numerical trajectories', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('RK8_initial_errors.png', dpi=500, bbox_inches='tight')
plt.show()


# Repeat with decreasing ∆t until no significant improvement
computation_time = []
time_vectors = {} 
positions_errors = {}
for delta_t in delta_t_values:
    # Numerical trajectory
    # 1. Initilie timer 
    start_time = time.time()

    # 2. Integration time interval
    t_eval = np.arange(0, tof, delta_t)

    # 3. Numerical trajectory
    sol = solve_ivp(two_body, t_span, w0, method='DOP853', t_eval=t_eval, rtol=1e-9, atol=1e-12)

    # 4. Stop timer 
    stop_time = time.time()

    # 5.Computational time
    computation_time.append(stop_time-start_time)
    print(f"RK8 Time of computation for {n_orbits} orbits with an integration interval of {delta_t} s: {stop_time-start_time} s")

    # 6. Satellite numerical position
    positions_num = sol.y[6:9, :].T 

    # Analytical trajectory
    # 1.Initial eccentric anomaly
    E_0 = utils.eccentric_anomaly(e_B,initial_true_anomaly)

    # 2. Initial mean anomaly 
    M_0 = utils.mean_anomaly_from_eccentric_anomaly(E_0,e_B)

    # 3. Mean motion
    n = utils.mean_motion(mu_earth,a_B)

    # 4. True anomaly at each instant of time 
    true_anomalies_exact = []
    t = 0 #Initial position
    for _ in range(positions_num.shape[0]):

        # 4.1. Mean anomaly  
        M = utils.mean_anomaly_from_mean_motion(M_0,n,t)

        # 4.2. Eccentric anomaly from mean anomaly
        E = utils.solve_Kepler(e_B, M)

        # 4.3. True anomaly
        true_anomalies_exact.append(utils.compute_true_anomaly_from_E(e_B,E))

        # 4.4 Update time 
        t = t + delta_t

    true_anomalies_exact = np.array(true_anomalies_exact)

    # 5. Radius at each point of the analytical trajectory 
    r_exact = []
    for theta in true_anomalies_exact:
        r_exact.append(utils.orbit_radius_from_anomaly(a_B,e_B,theta))

    r_exact = np.array(r_exact)

    # 6. Position at each true anomaly of the analytical trajectory
    positions_exact = np.array([utils.compute_position(r, theta) for r, theta in zip(r_exact, true_anomalies_exact)])

    # 7. Position error
    pos_error = np.linalg.norm((positions_num-positions_exact),axis=1)

    # 8. Time vector 
    time_vector = np.arange(0, T * n_orbits, delta_t)

    # 9. Save results 
    positions_errors[delta_t] = pos_error
    time_vectors[delta_t] = time_vector

# 10. Find the smallest delta t to interpolate he vectors corresponding to larger delta_t so that they have the same length of vector
dt_ref = min(delta_t_values)
time_ref = time_vectors[dt_ref]

delta_theta_interp = {}
for delta_t in delta_t_values:
    time_vector = time_vectors[delta_t]
    delta_pos = positions_errors[delta_t]

    # Interpolation to the minimum delta t
    delta_theta_interp[delta_t] = np.interp(time_ref, time_vector, delta_pos)


# Save results in a txt file
sys.stdout.close()
sys.stdout = sys.__stdout__


# Plot results 
# 1. Position error
n_plots = len(delta_theta_interp)
fig, axes = plt.subplots(n_plots, 1, figsize=(20, 3*n_plots), sharex=True) 

for ax, (delta_t, delta_pos_interp) in zip(axes, delta_theta_interp.items()):
    ax.plot(time_ref/3600, delta_pos_interp)
    ax.set_ylabel(f' Δp [m]')
    ax.set_title(f'Δt = {delta_t} s',fontweight='bold', y=0.8, fontsize=16)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

axes[-1].set_xlabel('Time [h]')
plt.tight_layout()
plt.savefig('RK8_error_evolution.png', dpi=500,  bbox_inches='tight')
plt.show()


# 2. Computational time
delta_t_strings = [str(x) for x in delta_t_values]
plt.figure(figsize=(6,5))
plt.scatter(delta_t_strings,computation_time, color = 'blue')
plt.plot(delta_t_strings, computation_time, linestyle='--', color='green', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('RK8 evolution of the computaitonal time with the time integration step', fontweight='bold', fontsize=12)
plt.xlabel('Δt [s]')
plt.ylabel('Time [s]')
plt.savefig('RK8_time_evolution.png', dpi=500,  bbox_inches='tight')
plt.show()


# Measure errors and especially time offsets in perigee/apogee passage versus exact solution with sufficiently small ∆t.
# 1. Integration for 100 orbits
n_orbits = 100
tof = T*n_orbits
t_span = (0, tof)

# 2. Solve for a small ∆t
t_eval = np.arange(0, tof, delta_t_values[-1]) # Smallest value
sol = solve_ivp(two_body, t_span, w0, method='DOP853', t_eval=t_eval, rtol=1e-9, atol=1e-12)
positions_num = sol.y[6:9, :].T  
r_solver = utils.orbit_radii_from_coordinates(positions_num)

# 3. Analytical solution
# 3.1. True anomaly at each instant of time 
true_anomalies_exact = []
t = 0 #Initial position
for _ in range(positions_num.shape[0]):

    # 3.1.1. Mean anomaly  
    M = utils.mean_anomaly_from_mean_motion(M_0,n,t)

    # 3.1.2. Eccentric anomaly from mean anomaly
    E = utils.solve_Kepler(e_B, M)

    # 3.1.3. True anomaly
    true_anomalies_exact.append(utils.compute_true_anomaly_from_E(e_B,E))

    # 3.1.4. Update time 
    t = t + delta_t

true_anomalies_exact = np.array(true_anomalies_exact)

# 3.2. Radius at each point of the analytical trajectory 
r_exact = []
for theta in true_anomalies_exact:
    r_exact.append(utils.orbit_radius_from_anomaly(a_B,e_B,theta))

r_exact = np.array(r_exact)

# 3.3. Position at each true anomaly of the analytical trajectory
positions_exact = np.array([utils.compute_position(r, theta) for r, theta in zip(r_exact, true_anomalies_exact)])

#4. Position error
pos_error = np.linalg.norm((positions_num-positions_exact),axis=1)

# 5. Time offsets 
# 5.1. Perigee
# 5.1.1. Find indexes of apogee at eacch orbit (this function finds local minimums and maximuns)
perigee_indices_num = argrelextrema(r_solver, np.less)[0]   
perigee_indices_exact = argrelextrema(r_exact, np.less)[0]

# 5.1.2. Compute offset
instant_time = np.arange(0,tof, delta_t_values[-1]) 
perigee_offsets = []
for idx_num, idx_exact in zip(perigee_indices_num, perigee_indices_exact):
    t_num = instant_time[idx_num]
    t_exact = instant_time[idx_exact]
    perigee_offsets.append(t_num - t_exact)

# 5.2. Apogee
# 5.2.1. Find indexes of apogee at eacch orbit (this function finds local minimums and maximuns)
apogee_indices_num  = argrelextrema(r_solver, np.greater)[0] 
apogee_indices_exact  = argrelextrema(r_exact, np.greater)[0]

# 5.2.2. Compute offset
apogee_offsets = []
for idx_num, idx_exact in zip(apogee_indices_num, apogee_indices_exact):
    t_num = instant_time[idx_num]
    t_exact = instant_time[idx_exact]
    apogee_offsets.append(t_num - t_exact)


# Plot results
# 1. Positions error
plt.plot(instant_time/3600, pos_error)
plt.ylabel(f' Δp [m]')
plt.xlabel('Time of flight [h]')
plt.title(f'RK8 positions error for Δt = {delta_t} s and {n_orbits} orbits',fontweight='bold', fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('RK8_pos_error.png', dpi=500,  bbox_inches='tight')
plt.show()

# 2. Perigee
orbit_numbers = np.arange(0,n_orbits)
orbit_numbers = orbit_numbers.tolist()
orbit_numbers_string = [str(x) for x in orbit_numbers]
min_len = min(len(orbit_numbers_string), len(perigee_offsets))
plt.figure(figsize=(10,6))
plt.scatter(orbit_numbers_string[:min_len],perigee_offsets[:min_len], color = 'blue')
plt.plot(orbit_numbers_string[:min_len],perigee_offsets[:min_len], linestyle='--', color='green', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=-90, fontsize=6)
plt.title(f'RK8 time offset in perigee for Δt = {delta_t_values[-1]} s and {n_orbits} orbits', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('Time offset [s]')
plt.savefig('RK8_time_offset_perigee.png', dpi=500,  bbox_inches='tight')
plt.show()

# 3. Apogee
plt.figure(figsize=(10,6))
min_len = min(len(orbit_numbers_string), len(perigee_offsets))
plt.scatter(orbit_numbers_string[:min_len],apogee_offsets[:min_len], color = 'blue')
plt.plot(orbit_numbers_string[:min_len],apogee_offsets[:min_len], linestyle='--', color='green', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=-90, fontsize=6)
plt.title(f'RK8 time offset in apogee for Δt = {delta_t_values[-1]} s and {n_orbits} orbits', fontweight='bold', fontsize=12)
plt.xlabel('# orbit')
plt.ylabel('Time offset [s]')
plt.savefig('RK8_time_offset_apogee.png', dpi=500,  bbox_inches='tight')
plt.show()


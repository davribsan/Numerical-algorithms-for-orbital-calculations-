'''
(a) Orbital Mechanics Solver based on Runge-Kutta 4th order
(b) Version number: 1
(c) Autors: davribsan
(d) Date of initializaition: 09/10/25
(e) Description of the program: Given an orbit, compute the exact solution of the central-body problem, 
and integrate the equations of motion of a satellite around a point-like Earth to study numerical 
convergence using a Runge-Kutta 4th order numerical solver. The problem is considered in only two dimensions.

(f) Numerical solver: Runnge-Kutta 4th order
    References:
    Cyber Omelette. (2017). N-Body Orbit Simulation with Runge-Kutta. 
    Url: https://www.cyber-omelette.com/2017/02/RK4.html
    
g) Range of validity expected of the parameters and range tested:
   - Expected: 5 orbits with e = 0.9 and a = 70000km and an integration step size of 100 s.
   - Tested: 5 orbits with e = 0.9 and a = 70000km and an integration step size of [100, 50, 20, 10, 5] s 
             and a spacecraft of 700 kg.

(h) Inputs:

    G_EARTH: Earth gravitational constant                            [m3⋅kg-1⋅s-2]
    M_EARTH: Mass of Earth (constant)                                [kg]
    R_EARTH: Earth radius (constant)                                 [m]
    mass_sat: Mass of the satellite                                  [kg] 
    a_B: Semimajor axis of the orbit from the problem assignament    [m]
    e_B: Eccentricity of the orbit from the probelm assignament      [ ]
    initial_true_anomaly: initial true anomaly                       [º]
    n_orbits: number of orbits to compute                            [ ]
    delta_t_values: list with integration step times                 [s]

(i) Outputs:
    Numerical and analytical trajectories
    Graph of radius/true anomaly/position error for fixed step integration time
    Graph of position erros for different intgration step times
    Grapgh pf computational time for different integration step times
    
(j) List of dependencies:
    - This program requires the Runge-Kutta 4th order numerical algorithm
    - This program requires the script "utils.py"

(k) Software version: Python 3.12.4
'''

# ------------------------------------------------------------------------------------
# Libraries 

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from auxiliar import rk4_solver
from auxiliar import utils

# ------------------------------------------------------------------------------------
# Constants

G_EARTH = 6.6743e-11 
M_EARTH = 5.97219e24 
R_EARTH = 6378e3

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
sys.stdout = open("RK4_results.txt", "w")

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

# 6. Join estates (because of the R-K 4 differential equation)
w0 = np.array(earth + sat)
m = np.array([M_EARTH, mass_sat]) 

# 7. Time of flight
T = utils.orbital_period(a_B,mu_earth)
tof = T*n_orbits
print('\nOrbital period:',T/(3600))

# 8. Initialize timer
start_time = time.time()

# 9. Numerical trajectory
x_solver, y_solver, z_solver, vx_solver, vy_solver, vz_solver = rk4_solver.RK4(tof, rk4_solver.fast, w0, m, delta_t_values[0])

# 10. Stop timer
stop_time = time.time()
print(f"RK4 Time of computation for {n_orbits} orbits with an integration interval of {delta_t_values[0]} s: {stop_time-start_time} s")

# 11. Satellite numerical position
x_solver = x_solver[:,1]
y_solver = y_solver[:,1]
z_solver = z_solver[:,1]
positions_num = np.column_stack((x_solver, y_solver, z_solver))

# 12. Satellite numerical velocity
vx_solver = vx_solver[:,1]
vy_solver = vy_solver[:,1]
vz_solver = vz_solver[:,1]
velocities_num = np.column_stack((vx_solver, vy_solver, vz_solver))

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

plt.title('RK4 Numerical and Analytical trajectories', fontweight='bold', fontsize=12)
plt.xlabel('X[m]')
plt.ylabel('Y[m]')
plt.legend(loc = 'upper right')
plt.axis('equal')
plt.savefig('RK4_trajectories.png', dpi=500, bbox_inches='tight')
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
ax1.set_title('RK4 radii difference between analytical and numerical trajectories', fontweight='bold', fontsize=14)

# 4.2. Numerical-analytical true anomaly diference 
max_anomaly_error = 20
n_outliers = np.sum(delta_theta > max_anomaly_error) # Discard outliers for representation purposes
delta_theta[delta_theta > max_anomaly_error] = 0
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
ax2.scatter(instant_time/3600, delta_theta, color='dodgerblue', s=1)
ax2.set_ylabel('Δθ [º]')
ax2.text(0.01, 0.8, f'Discarding {n_outliers} outliers > {max_anomaly_error}º for representation effects',
         fontsize=12, color='red', ha='left', va='bottom', transform=ax2.transAxes)
ax2.set_title('RK4 true anomaly difference between analytical and numerical trajectories', fontweight='bold', fontsize=14)

# 4.3. Numerical-analytical position diference  
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
ax3.scatter(instant_time/3600, pos_error, color='dodgerblue', s=1)
ax3.set_ylabel('Δp [m]')
ax3.set_xlabel('Time of flight [h]')
ax3.set_title('RK4 position error between analytical and numerical trajectories', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('RK4_initial_errors.png', dpi=500, bbox_inches='tight')
plt.show()


# Repeat with decreasing ∆t until no significant improvement
computation_time = []
time_vectors = {} 
positions_errors = {}

for delta_t in delta_t_values:
    # Numerical trajectory
    # 1.1. Initialize timer 
    start_time = time.time()

    # 1.2. Numerical trajectory
    x_solver, y_solver, z_solver, vx_solver, vy_solver, vz_solver = rk4_solver.RK4(tof, rk4_solver.fast, w0, m, delta_t)

    # 1.3. Stop timer 
    stop_time = time.time()

    # 1.4.Computational time
    computation_time.append(stop_time-start_time)
    print(f"RK4 Time of computation for {n_orbits} orbits with an integration interval of {delta_t} s: {stop_time-start_time} s")

    # 1.5. Satellite numerical position
    x_solver = x_solver[:,1]
    y_solver = y_solver[:,1]
    z_solver = z_solver[:,1]
    positions_num = np.column_stack((x_solver, y_solver, z_solver))

    # Analytical trajectory
    # 2.1.Initial eccentric anomaly
    E_0 = utils.eccentric_anomaly(e_B,initial_true_anomaly)

    # 2.2. Initial mean anomaly 
    M_0 = utils.mean_anomaly_from_eccentric_anomaly(E_0,e_B)

    # 2.3. Mean motion
    n = utils.mean_motion(mu_earth,a_B)

    # 2.4. True anomaly at each instant of time 
    true_anomalies_exact = []
    t = 0 #Initial position
    for _ in range(positions_num.shape[0]):

        # 2.4.1. Mean anomaly  
        M = utils.mean_anomaly_from_mean_motion(M_0,n,t)

        # 2.4.2. Eccentric anomaly from mean anomaly
        E = utils.solve_Kepler(e_B, M)

        # 2.4.3. True anomaly
        true_anomalies_exact.append(utils.compute_true_anomaly_from_E(e_B,E))

        # 2.4.4 Update time 
        t = t + delta_t

    true_anomalies_exact = np.array(true_anomalies_exact)

    # 2.5. Radius at each point of the analytical trajectory 
    r_exact = []
    for theta in true_anomalies_exact:
        r_exact.append(utils.orbit_radius_from_anomaly(a_B,e_B,theta))

    r_exact = np.array(r_exact)

    # 2.6. Position at each true anomaly of the analytical trajectory
    positions_exact = np.array([utils.compute_position(r, theta) for r, theta in zip(r_exact, true_anomalies_exact)])

    # 3. Position error
    pos_error = np.linalg.norm((positions_num-positions_exact),axis=1)

    # 4. Time vector 
    time_vector = np.arange(0, T * n_orbits, delta_t)

    # 5. Save results 
    positions_errors[delta_t] = pos_error
    time_vectors[delta_t] = time_vector

# 6. Find the smallest delta t to interpolate he vectors corresponding to larger delta_t so that they have the same length of vector
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
plt.savefig('RK4_error_evolution.png', dpi=500,  bbox_inches='tight')
plt.show()

# 2. Computational time
delta_t_strings = [str(x) for x in delta_t_values]
plt.figure(figsize=(6,5))
plt.scatter(delta_t_strings,computation_time, color = 'blue')
plt.plot(delta_t_strings, computation_time, linestyle='--', color='green', alpha=0.7)
y_min = 0
y_max = max(computation_time) + 5
plt.yticks(np.arange(y_min, y_max+1, 5)) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('RK4 evolution of the computaitonal time with the time integration step', fontweight='bold', fontsize=12)
plt.xlabel('Δt [s]')
plt.ylabel('Time [s]')
plt.savefig('RK4_time_evolution.png', dpi=500,  bbox_inches='tight')
plt.show()










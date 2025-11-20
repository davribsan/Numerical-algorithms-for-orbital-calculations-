'''
(a) Lambert's Problem
(b) Version number: 1
(c) Autors: davribsan
(d) Date of initializaition: 29/09/25
(e) Description of the program: 
    1. Using the equations of orbital mechanics, determination of the time of flight (tof) between two given 
       true anomalies, θ1 and θ2 (∆θ), both the initial and final radii, r1 and r2, as well as the respective
       velocities at those points, v1 and v2 for a 2D scenario around the Earth.  
    2. Solution of Lambert's problem using previous r1, r2, tof, and ∆θ to determine the original orbit, that is, 
       determine: v1, v2, the initial and final true anomalies θ1,θ2, and a and e.
    3. Find the best configuration of the solver's paramters to achieve the best result. Compute the perigee
       of the orbit and altitude at that point. It has been considered only the short path as well as no revolutions.
  
(f) Lambert sover: Izzo 
    References:
    [1] Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics
           and Dynamical Astronomy, 121(1), 1-15.

    [2] Lancaster, E. R., & Blanchard, R. C. (1969). A unified form of
           Lambert's theorem (Vol. 5368). National Aeronautics and Space
           Administration.

g) Range of validity expected of the parameters and range tested:
   - Expected: An average error in the velocity vector of 10^-13 is expected, with a maximum error of 10-8.
   For the single revolution case, the expected average of iterations is 2.1 iterations while, 
   in the multiple revolution case 3.3. No convergence when the number of iterations is too small.
   - Tested: It has been tested for an orbit with a = 7000 km and e = 0.1.  

(h) Inputs:

    G_EARTH: Earth gravitational constant                            [m3⋅kg-1⋅s-2]
    M_EARTH: Mass of Earth (constant)                                [kg]
    R_EARTH: Earth radius (constant)                                 [m]
    theta_1: true anomaly 1 from the problem assignament             [º]
    theta_2: true anomaly 2 from the problem assignament             [º]
    a_A: semimajor axis of the orbit from the problem assignament    [m]
    e_A: eccentricity of the orbit from the probelm assignament      [ ]
    maxiter: number of iterations the solver will do                 [ ]
    atol: absolute tolerance of the solver                           [ ]
    rtol: relative tolerance of the solver                           [ ]

(i) Outputs:
    
    r1: radius at initial true anomaly                               [m]
    r2: radius at final true anomaly                                 [m]
    tof: time of flight between true anomalies                       [s]
    v1_theoretical: velocity in p1 from Kepler equations             [m/s]
    v2_theoretical: velocity in p2 from Kepler equations             [m/s]
    v1: velocity in p1 from Lambert's solution                       [m/s]
    v2: velocity in p2 from Lambert's solution                       [m/s]
    a: semimajor axis of the orbit from Lambert's solution           [m]
    e: eccentricity of the orbit from Lambert's solution             [ ]
    true_a_1: final true anomaly for point 1 from Lambert's solution [º]
    true_a_1: final true anomaly for point 2 from Lambert's solution [º]
    a_error: relative error of the semimajor axis of the orbit       [%]
    e_error: relative error of the eccentricity of the orbit         [%]
    true_a_1_error: absolute error of the initial true anomaly       [º]  
    true_a_2_error: absolute error of the final true anomaly         [º]
    
(j) List of dependencies:
    - This program requires the izzo Lambert solver
    - This program requires the script "utils.py"

(k) Software version: Python 3.12.4

'''

# ------------------------------------------------------------------------------------
# Libraries 

import numpy as np
import sys
from auxiliar import izzo
from auxiliar import utils

# ------------------------------------------------------------------------------------
# Constants

G_EARTH = 6.6743e-11 
M_EARTH = 5.97219e24 
R_EARTH = 6378e3

# ------------------------------------------------------------------------------------
# Inputs 

theta_1 = 30                  
theta_2 = 60
a_A = 7000e3              
e_A = 0.1                 
maxiter = 50              
atol = 1e-7                
rtol = 1e-7                

# ------------------------------------------------------------------------------------
# Program

# 0. Save all results printed on the screen in a file 
sys.stdout = open("Lambert_solution_results.txt", "w")

# 1. Kepler equations
# 1.1. Standard gravitational parameter 
mu_earth = G_EARTH*M_EARTH 

# 1.2. Radius of the true anomalies 
r1 = utils.orbit_radius_from_anomaly(a_A, e_A, theta_1)
r2 = utils.orbit_radius_from_anomaly(a_A, e_A, theta_2)
print(f"\nThe initial radius at true anomaly 1 ({theta_1}º): {r1} m")
print(f"The final radius at true anomaly 2 ({theta_2}º): {r2} m")

# 1.3. Velocity at true anomalies 
v1_theoretical = utils.orbit_velocity(mu_earth,a_A,r1)
v2_theoretical = utils.orbit_velocity(mu_earth,a_A,r2)
print(f"\nInitial velocity from Kepler equations: {v1_theoretical} m/s")
print(f"Final velocity from Kepler equations: {v2_theoretical} m/s")

# 1.3.Eccentric anomaly
E1 = utils.eccentric_anomaly(e_A,theta_1)
E2 = utils.eccentric_anomaly(e_A,theta_2)

# 1.4. Mean anomaly
M1 = utils.mean_anomaly(E1,e_A)
M2 = utils.mean_anomaly(E2,e_A)

# 1.5. Mean motion
n = utils.mean_motion(mu_earth,a_A)

# 1.6. Time of flight
tof = utils.time_of_flight(M1,M2,n)
print(f"\nTime of flight between true anomalies: {tof} s")


# 2. Solve Lambert's problem
p1 = utils.compute_position(r1,theta_1) 
p2 = utils.compute_position(r2,theta_2) 

v1_, v2_ = izzo.izzo2015(mu_earth, p1, p2, tof, maxiter, atol, rtol)

v1 = np.linalg.norm(v1_)
v2 = np.linalg.norm(v2_)

print(f"\nInitial velocity from Lambert's solution: {v1} m/s")
print(f"Final velocity from Lambert's solution: {v2} m/s")

# 2.1. Semimajor axis 
a = utils.compute_semimajor_axis(r1,v1,mu_earth)
a_error = utils.relative_error(a_A,a)

print(f"\nSemimajor axis from Lambert's solution: {a} m")
print(f"Relative error: {a_error} %")

# 2.2. Eccentricity
e = utils.compute_eccentricity(p1,v1_,mu_earth) 
e_error = utils.relative_error(e_A,e)

print("\nEccentricity from Lambert's solution: ",e) 
print(f"Relative error: {e_error} %")

# 2.3. True anomalies 
true_a_1 = utils.compute_true_anomaly(a,e,r1)
true_a_2 = utils.compute_true_anomaly(a,e,r2)

true_a_1_error = utils.anomalies_error(theta_1,true_a_1)
true_a_2_error = utils.anomalies_error(theta_2,true_a_2)

print(f"\nInitial true anomaly from Lambert's solution: {true_a_1} º")
print(f"Absolute error: {true_a_1_error} º")

print(f"Final true anomaly from Lambert's solution: {true_a_2} º")
print(f"Absolute error: {true_a_2_error} º")


# 3.1. Find the optimal configuration of parameters of the solver 
tolerances = np.logspace(-1, -13, num=13, base=10) # Array with tolerances ranging from 10^-1 to 10^-13
max_iterations = 100  # Maximum number of iteratons for the solver 
initial_iteration = 3 # Minimum number of iterations that allows the solver to converge for a tolerance of 10^-1

best_combinations = np.zeros((3,tolerances.shape[0])) # Array to save the best combinations 
tolerance_index = 0 # Auxiliar variable to use during iterations 

for tolerance in tolerances:

    # 3.1.1. Lambert's solution
    atol = tolerance
    rtol = tolerance
    v1_, v2_ = izzo.izzo2015(mu_earth, p1, p2, tof, initial_iteration, atol, rtol)
    v1 = np.linalg.norm(v1_)
    v2 = np.linalg.norm(v2_)

    # 3.1.2. Semimajor axis 
    a = utils.compute_semimajor_axis(r1,v1,mu_earth)
    a_error = utils.relative_error(a_A,a)

    # 3.1.3. Eccentricity
    e = utils.compute_eccentricity(p1,v1_,mu_earth)
    e_error = utils.relative_error(e_A,e)

    # 3.1.4. Compute the error of the solver 
    combination_error = utils.root_mean_square_error(a_error,e_error)
    best_combinations[0,tolerance_index] = tolerance
    best_combinations[1,tolerance_index] = initial_iteration
    best_combinations[2,tolerance_index] = combination_error

    for n_iterations in range(initial_iteration+1,max_iterations+initial_iteration):
        # 3.1.5. Lambert's solution
        maxiter = n_iterations
        atol = tolerance
        rtol = tolerance
        v1_, v2_ = izzo.izzo2015(mu_earth, p1, p2, tof, maxiter, atol, rtol)
        v1 = np.linalg.norm(v1_)
        v2 = np.linalg.norm(v2_)
        
        # 3.1.6. Semimajor axis 
        a = utils.compute_semimajor_axis(r1,v1,mu_earth)
        a_error = utils.relative_error(a_A,a)

        # 3.1.7. Eccentricity
        e = utils.compute_eccentricity(p1,v1_,mu_earth)
        e_error = utils.relative_error(e_A,e)

        # 3.1.8. Compute the error of the solver 
        iteration_error = utils.root_mean_square_error(a_error,e_error)
        
        # 3.1.9. If there is an error smaller than the previous minimum, update it 
        if iteration_error < combination_error:
            combination_error = iteration_error
            best_combinations[0,tolerance_index] = tolerance
            best_combinations[1,tolerance_index] = n_iterations
            best_combinations[2,tolerance_index] = combination_error

    tolerance_index = tolerance_index + 1

# 3.1.10. Find the minimum value among all the combinations
min_index = np.argmin(best_combinations[2]) # Search for the minimum error 
min_value = best_combinations[2, min_index] # Combination whose error belongs to 
print('\nBest configuration:\n  Tolerance: ', best_combinations[0,min_index])
print('  Number of iterations:', int(best_combinations[1,min_index]))

# 3.2. Solve for the best combination of parameters
# 3.2.1. Solve Lamberts problem
maxiter = best_combinations[1,min_index] # Optimal number of iterations
atol = best_combinations[0,min_index]    # Best absolute tolerance
rtol = best_combinations[0,min_index]    # Best relative tolerance
v1_, v2_ = izzo.izzo2015(mu_earth, p1, p2, tof, maxiter, atol, rtol)

v1 = np.linalg.norm(v1_)
v2 = np.linalg.norm(v2_)

print('\nBest solution:')
print(f" Initial velocity from Lambert's solution: {v1} m/s")
print(f" Final velocity from Lambert's solution: {v2} m/s")

# 3.2.2. Semimajor axis 
a = utils.compute_semimajor_axis(r1,v1,mu_earth)
a_error = utils.relative_error(a_A,a)

print(f"\n Semimajor axis from Lambert's solution: {a} m")
print(f" Relative error: {a_error} %")

# 3.2.3. Eccentricity
e = utils.compute_eccentricity(p1,v1_,mu_earth)
e_error = utils.relative_error(e_A,e)

print("\n Eccentricity from Lambert's solution: ",e) 
print(f" Relative error: {e_error} %")

# 3.2.4. True anomalies 
true_a_1 = utils.compute_true_anomaly(a,e,r1)
true_a_2 = utils.compute_true_anomaly(a,e,r2)

true_a_1_error = utils.anomalies_error(theta_1,true_a_1)
true_a_2_error = utils.anomalies_error(theta_2,true_a_2)

print(f"\n Initial true anomaly from Lambert's solution: {true_a_1} º")
print(f" Absolute error {true_a_1_error} º")

print(f" Final true anomaly from Lambert's solution: {true_a_2} º")
print(f" Absolute error: {true_a_2_error} º")


# 4. Save results in a txt file
sys.stdout.close()
sys.stdout = sys.__stdout__
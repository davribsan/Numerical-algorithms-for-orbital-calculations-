'''
(a) Pork-chop diagram
(b) Version number: 1
(c) Autors: davribsan
(d) Date of initializaition: 02/10/25
(e) Description of the program: 
    Using the circular orbits approximation in the same plane, show the pork-chop diagram 
    for a transfer from a circular LEO at 200 km altitude to the Moon, considering the orbit 
    of the Moon circular (using the average distance) and neglecting the Moon's attraction. 
    For visual reasons, the tof instead of the arrival dates are shown.    
(f) Lambert sover: Izzo 
    References:
    [1] Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics
           and Dynamical Astronomy, 121(1), 1-15.

    [2] Lancaster, E. R., & Blanchard, R. C. (1969). A unified form of
           Lambert's theorem (Vol. 5368). National Aeronautics and Space
           Administration.

g) Range of validity expected of the parameters and range tested
    - Expected: It was expected to work for circular orbits with a minimum time of flight = 3 days.
    - Tested: Time of flights from 3 to 15 days.

(h) Inputs:

    G_EARTH: Earth gravitational constant                       [m3⋅kg-1⋅s-2]
    M_EARTH: Mass of Earth (constant)                           [kg]
    R_EARTH: Earth radius (constant)                            [m]
    r_MOON: mean distance Moon-Earth                            [m]
    h_LEO: altitude of the orbot                                [m]
    maxiter: number of iterations the solver will do            [ ]
    atol: absolute tolerance of the solver                      [ ]
    rtol: relative tolerance of the solver                      [ ]

(i) Outputs:
    Pork-chop diagram
    Orbits + Moon position + Spacecraft in LEO position (example for verification)
    
(j) List of dependencies:
    - This program requires the izzo Lambert solver.
    - This program requires the script "utils.py"

(k) Software version: Python 3.12.4
'''

# ------------------------------------------------------------------------------------
# Libraries 

import numpy as np
import matplotlib.pyplot as plt
from auxiliar import izzo
from auxiliar import utils

# ------------------------------------------------------------------------------------
# Constants

G_EARTH = 6.6743e-11 
M_EARTH = 5.97219e24 
R_EARTH = 6378e3
r_MOON = 384000e3 

# ------------------------------------------------------------------------------------
# Inputs 

h_LEO = 200000
maxiter = 5              
atol = 1e-10                
rtol = 1e-10                

# ------------------------------------------------------------------------------------
# Program

# 1. Standard gravitational parameter 
mu_earth = G_EARTH*M_EARTH 

# 2. Altitude of the LEO orbit from the centre of the Earth
r_LEO = R_EARTH + h_LEO 

# 3. Orbits angular velocities 
omega_LEO = utils.angular_velocity(mu_earth,r_LEO)      # rad/s
omega_moon = utils.angular_velocity(mu_earth,r_MOON)    # rad/s

# 4. Time windows in days
departure_days = np.arange(0, 30, 1)       
tof_days = np.arange(3, 15, 1) 

delta_V = np.zeros((len(tof_days), len(departure_days))) # Array to store the delta V for each combination 

# 5. Compute all possible combinations
for i, t_depart in enumerate(departure_days):
    for j, tof in enumerate(tof_days):
        
        # 5.1. Compute the moment of arrival and the time of flight 
        t_arrive = t_depart + tof
        delta_t  = utils.days_to_seconds(tof)
        
        # 5.2. Compute the position in the circular orbit at a certain instant of time 
        p1 = utils.circular_position(r_LEO, omega_LEO, utils.days_to_seconds(t_depart))
        p2 = utils.circular_position(r_MOON, omega_moon, utils.days_to_seconds(t_arrive))

        # 5.3. Compute the insertion/exit velocity from one orbit to the other 
        v1_, v2_ = izzo.izzo2015(mu_earth, p1, p2, delta_t, maxiter, atol, rtol)

        # 5.4. Compute the orbital velocities 
        v_LEO = utils.circular_velocity(r_LEO, omega_LEO, utils.days_to_seconds(t_depart))
        v_Moon = utils.circular_velocity(r_MOON, omega_moon, utils.days_to_seconds(t_arrive))

        # 5.5. Compute the total delta V
        delta_V[j,i] = np.linalg.norm(v1_ - v_LEO) + np.linalg.norm(v2_ - v_Moon)

 
# 6. Create a grid with all possible combinations departure-arrival
dep, tof = np.meshgrid(departure_days, tof_days)

# 7. Plot the pork-chop diagram
plt.figure(figsize=(6, 5))
levels = np.linspace(delta_V.min(), delta_V.max(), 25) 

# 7.1. Draw the contour of the digram "ellipses"
cs = plt.contour(dep, tof, delta_V/1000, levels=levels/1000, colors='white', linewidths=0.5) 

# 7.2. Color the regions pf the diagram
cp = plt.contourf(dep, tof, delta_V/1000, levels=levels/1000, cmap='viridis') 

# 7.3. Legend of the delta-v "intensity"
cbar = plt.colorbar(cp) 

# 7.4. Write axes labels 
cbar.set_label('Total delta-V (km/s)')
plt.xlabel('Departure Day')
plt.ylabel('Time of flight') # If we want a real arrival date in the diagram it'd look as a half "representation"   
plt.title('Pork-Chop LEO - Moon')
plt.savefig('Pork_chop_diagram.png', dpi=500,  bbox_inches='tight')
plt.show()


'''
# Example for verification

t_depart = 11                  # Departure day
tof = 7                        # Time of flight in days
t_arrive = t_depart + tof      # Time of arrival
delta_t = days_to_seconds(tof)

# Compute the departue/arrival positions in each orbit
p1 = circular_position(r_LEO, omega_LEO, t_depart*24*3600)
p2 = circular_position(r_MOON, omega_moon, t_arrive*24*3600)

# Compute the exit/injection velocities from one orbit to the other
v1_, v2_ = izzo.izzo2015(mu_earth, p1, p2, delta_t, maxiter, atol, rtol)

# Sketch the positions  
plt.figure(figsize=(8,8))

# Plot circular orbits
theta_circle = np.linspace(0, 2*np.pi, 300)
plt.plot(r_LEO*np.cos(theta_circle), r_LEO*np.sin(theta_circle), linestyle = '--', linewidth=0.5, color='black', label='LEO')
plt.plot(r_MOON*np.cos(theta_circle), r_MOON*np.sin(theta_circle), linestyle = '--', linewidth=0.5, color='black', label='Lunar orbit')

# Plot Earth
plt.plot(0,0, marker='o', markersize=5, color = 'blue', label='Earth')

# Plot satellite initial position 
plt.plot(p1[0], p1[1], marker='s', color = 'red', markersize=3, label='Aircraft in LEO')

# Plot Moon's position at arrival date
plt.plot(p2[0], p2[1], marker='o', markersize=5, color = 'grey', label='Moon')

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Moon-Aircraft positions')
plt.show()'''

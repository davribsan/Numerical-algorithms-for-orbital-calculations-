'''
(a)Auxiliary functions for orbital calculations
(b) Version number: 1
(c) Autors: davribsan
(d) Date of initializaition: 29/09/25
(e) Description of the program: Functions for orbital calculations
(f) List of dependencies: This script was created with basic python libraries
(g) Software version: Python 3.12.4
'''

# ------------------------------------------------------------------------------------
# Libraries 

import numpy as np
import math

# ------------------------------------------------------------------------------------
# Functions

def circular_position(r, omega, t):
    "Computes the position in the circular orbit at a specific time"
    theta = (omega * t) % (2*math.pi) # The angle must be in the range 0-2pi
    return np.array([r * math.cos(theta), r * math.sin(theta), 0.0])

def angular_velocity(mu,r):
    "Computes the angular velocity of a circular orbit"
    return math.sqrt(mu/(r**3))

def circular_velocity(r, omega, t):
    "Computes the linear velocity of a circular orbit"
    theta = (omega * t) % (2*math.pi) # The angle must be in the range 0-2pi
    return np.array([-r * omega * math.sin(theta), r * omega * math.cos(theta), 0.0])

def days_to_seconds(days):
    "Converts time in days to seconds"
    return days*24*3600

def orbit_velocity(mu,a,r):
    "Computes the Keplerian orbital velocity at given point"
    v = math.sqrt(mu*(2/r - 1/a))
    return v

def eccentric_anomaly(e,theta):
    "Computes the eccentric anomaly in a Keplerian orbit for a given true anomaly"
    theta = math.radians(theta)
    E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(theta/2),
                         math.sqrt(1 + e) * math.cos(theta/2)) #Equivalent to tan but considering all the angles between +-180º
    return E

def mean_anomaly(E,e):
    "Computes the mean anomaly in a Keplerain orbit corresponding to certain true anomaly"
    M = E-e*math.sin(E)
    return M

def mean_motion(mu,a):
    "Computes the mean motion in a Keplerian orbit"
    n = math.sqrt(mu/(a**3))
    return n

def time_of_flight(M1,M2,n):
    "Time of flight between true anomalies"
    tof = (M2-M1)/n
    return tof
    
def compute_position(r,theta):
    "Computes the position in an orbit given the radius and true anomaly"
    x = r*math.cos(math.radians(theta))
    y = r*math.sin(math.radians(theta))
    z = 0
    p = np.array([x, y, z])
    return p.copy() 

def compute_semimajor_axis(r,v,mu):
    "Computes the semimajor axis of the orbit"
    a = (r*mu)/(2*mu-r*(v**2))
    return a

def eccentricity_vector(mu, p,v_):
    "Computes the eccentricity vector"
    # Velocity 
    v = np.linalg.norm(v_)

    # Radius 
    r = np.linalg.norm(p)

    # Eccentricity vector
    r_dot_v = np.dot(p, v_)
    e_vec = (1/mu) * (((v**2 - mu/r) * p) - r_dot_v * v_)
    return e_vec

def compute_eccentricity(p,v_,mu):
    "Computes the orbit eccentricity"
    # Eccentricity vector
    e_vec = eccentricity_vector(mu, p,v_)
    e = np.linalg.norm(e_vec)
    return e

def compute_true_anomaly(a,e,r):
    "Computes the true anomaly given the orbit parameters and the radius of the point"
    true_a = math.degrees(math.acos((1/e)*((a*(1-(e**2))/r)-1)))
    return true_a

def relative_error(true_value, measured_value):
    "Computes the relative error for the eccentricity and the major semiaxis"
    error = abs((measured_value - true_value)/true_value)
    return error*100

def anomalies_error(theoretical_theta, measured_theta):
    "Computes the absolute error of the true anomalies"
    theoretical_theta = math.radians(theoretical_theta)
    measured_theta = math.radians(measured_theta)
    tof = min(abs(measured_theta - theoretical_theta), 2*np.pi - abs(measured_theta - theoretical_theta))
    return math.degrees(tof)

def root_mean_square_error(a_error,e_error):
    "Computes the root mean square error of a and e to use as minimizing function"
    RMSE = math.sqrt((a_error**2 + e_error**2)/2)
    return RMSE

def orbit_radius_from_anomaly(a,e,theta):
    "Computes the radius of a point in the orbit given the true anomaly"
    r =  a * (1-e**2)/(1+e*math.cos(math.radians(theta)))
    return r

def orbit_radii_from_coordinates(positions):
    "Computes all the radii corresponding to each coordenate in the orbit"
    r = np.linalg.norm(positions, axis=1) 
    return r

def velocity_components(a,e,mu,theta,r):
    "Computes the velocity components in the three axes of the orbit"
    v_ = np.zeros((3))
    theta = math.radians(theta)

    # Specific angular momentum
    h = math.sqrt(mu*a*(1-e**2))

    # Radial velocity
    v_radial = (mu/h)*e*math.sin(theta)
    
    # Tangencial velocity
    v_tan = h/r

    # Velocity components in x,y
    v_[0] = v_radial*math.cos(theta) - v_tan*math.sin(theta)
    v_[1] = v_radial*math.sin(theta) + v_tan*math.cos(theta)
    v_[2] = 0
    return v_.copy()

def orbital_period(a,mu):
    "Computes the orbital period"
    T = 2*math.pi*math.sqrt(a**3/mu)
    return T

def mean_anomaly_from_eccentric_anomaly(E,e):
    "Computes the mean anomaly in a Keplerain orbit from the eccentric one"
    M = E-e*math.sin(E)
    M = (M + 2*math.pi) % (2*math.pi)
    return M

def mean_anomaly_from_mean_motion(M0,n,delta_t):
    "Computes the mean anomaly in a Keplerain orbit from the mean motion"
    M = (M0 + n*delta_t) % (2.0 * math.pi)
    return M

def solve_Kepler(e,M):
    "Newton-Rhapson method to solve Kepler's equation for eccentric anomaly E given mean anomaly M and eccentricity e"
    max_iter=100
    tol=1e-10
    E = M if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = E - e*math.sin(E) - M
        fp = 1 - e*math.cos(E)
        delta = f/fp
        E -= delta
        if abs(delta) < tol:
            break
    return E

def compute_true_anomaly_from_E(e,E):
    "Computes true anomaly from eccentric anomaly"
    theta =  2 * math.atan2(math.sqrt(1 + e) * math.sin(E/2),
                            math.sqrt(1 - e) * math.cos(E/2))
    return math.degrees(theta)

def compute_true_anomaly_from_r(mu,p,v_):
    "Computes the true anomaly given the orbit parameters and the radius of the point"
    # Eccentricity vector
    e_vec = eccentricity_vector(mu, p,v_)
    e = np.linalg.norm(e_vec)

    # True anomaly from atan2 
    cos_theta = np.dot(e_vec, p) / (e * np.linalg.norm(p))
    sin_theta = np.cross(e_vec, p)[2] / (e * np.linalg.norm(p))
    true_anomaly = np.degrees(np.arctan2(sin_theta, cos_theta))
    true_anomaly = np.mod(true_anomaly, 360)

    return true_anomaly

def specific_angular_momentum(p,v_):
    "Computes the specific angular momentum"
    h = np.cross(p,v_)
    return h

def nodal_vector(h):
    "Computes the nodal vector"
    n = np.array((-h[1],h[0],0))
    return n.copy()

def inclination(p,v_):
    "Computes the orbital inclination"
    # Specific angular momentum 
    h = specific_angular_momentum(p,v_)

    # Inclination
    i = math.acos(h[-1]/np.linalg.norm(h))
    return math.degrees(i)

def raan(p,v_):
    "Computes the right ascension of the acscending node"
    # Specific angular momentum 
    h = specific_angular_momentum(p,v_)

    # Nodal vector
    n = nodal_vector(h)

    # Right ascension of the acscending node
    omega = math.acos(n[0]/np.linalg.norm(n))

    if n[1]<0:
        omega = 2*math.pi - omega
    
    return math.degrees(omega)

def argumet_periapsis(mu,p,v_):
    "Computes the argument of the periapsis"
    # Specific angular momentum 
    h = specific_angular_momentum(p,v_)

    # Nodal vector
    n = nodal_vector(h)

    # Eccentricity vector
    e_vec = eccentricity_vector(mu, p,v_)

    # Argument of periapsis
    w = math.acos(np.dot(n,e_vec)/(np.linalg.norm(n)*np.linalg.norm(e_vec)))
    if e_vec[2]<0:
        w = 2*math.pi - w
    
    return math.degrees(w)
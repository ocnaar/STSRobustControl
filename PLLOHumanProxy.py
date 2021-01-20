# -*- coding: utf-8 -*-

"""

@author: Octavio Narvaez-Aroche ocnaar@berkeley.edu                                        
Berkeley Center for Control and Identification                             
Fall 2018                                              

Simulation of the Iterative Learning Control (ILC) algorithm used as a proxy 
for the action of the user of a minimally actuated powered lower limb orthosis 
(PLLO) while performing a sit-to-stand (STS) movement. 

The ILC algorithm is published in:
O. Narvaez Aroche, P. -J. Meyer, S. Tu, A. Packard and M. Arcak, "Robust 
Control of the Sit-to-Stand Movement for a Powered Lower Limb Orthosis," in 
IEEE Transactions on Control Systems Technology, vol. 28, no. 6, pp. 2390-2403,
Nov. 2020. http://dx.doi.org/10.1109/TCST.2019.2945908

Please cite our work accordingly.

This script uses the OpenAI Gym toolkit v0.17.3, please add it to your Anaconda 
installation before execution by following the instructions at:
https://anaconda.org/conda-forge/gym 

"""

# Import installed packages and modules. 
import gym
import gym.spaces
import numpy as np
import scipy
import scipy.linalg
import scipy.integrate
from scipy.io import loadmat

# Enable Numba for just-in-time (JIT) compilation of code with NumPy arrays and functions.  
DISABLE_JIT=False

if not DISABLE_JIT:
    print("NOTE: JIT enabled")

# Define jit decorator for using Numba.  
def _noop(func):
    return func

def jitwrapper(*args, **kws):
    if DISABLE_JIT:
        return _noop
    else:
        from numba import jit
        return jit(*args, **kws)

# Compile the function below in Numba without involving the Python interpreter.
@jitwrapper(nopython=True) 
def StateToOutput(p, x):
    """
    This function maps the state x of the three-link planar robot with revolute 
    joints that models the dynamics of the PLLO to the output y that it is 
    measured by the user while performing the STS movement.
    
    Arguments
    
    p: Parameters of the three-link robot model (unknown to the controller).
      p[0]: Mass of link 1 in [kg].
      p[1]: Mass of link 2 in [kg].
      p[2]: Mass of link 3 in [kg].
      p[3]: Moment of inertia of link 1 about its Center of Mass (CoM) in [kg.m^2].
      p[4]: Moment of inertia of link 2 about its CoM in [kg.m^2].
      p[5]: Moment of inertia of link 3 about its CoM in [kg.m^2].
      p[6]: Length of link 1 in [m].
      p[7]: Length of link 2 in [m].
      p[8]: Length of link 3 in [m].
      p[9]: Distance from ankle joint to CoM of link 1 in [m].
      p[10]: Distance from knee joint to CoM of link 2 in [m].
      p[11]: Distance from hip joint to CoM of link 3 in [m].
      
    x: State of the three-link robot. 
      x[0]: angular position of link 1 relative to the horizontal in [rad].
      x[1]: angular position of link 2 relative to link 1 in [rad].
      x[2]: angular position of link 3 relative to link 2 in [rad].
      x[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      x[4]: angular velocity of link 2 in [rad/s].
      x[5]: angular velocity of link 3 in [rad/s].

    Output

    y: Measurements performed by the user.
      y[0]: angular position of link 3 relative to link 2 in [rad].
      y[1]: x coordinate of the CoM in [m].
      y[2]: y coordinate of the CoM in [m].
      y[3]: angular velocity of link 3 relative to link 2 [rad/s].
      y[4]: x velocity of the CoM in [m/s].
      y[5]: y velocity of the CoM in [m/s].
      y[6]: current simulation time of the system in [s].
    """

    # Parameters of the system for computing the kinematics of the CoM.
    m1 = p[0]     # Mass of link 1 [kg].
    m2 = p[1]     # Mass of link 2 [kg].
    m3 = p[2]     # Mass of link 3 [kg].
    l1 = p[6]     # Length of link 1 [m].
    l2 = p[7]     # Length of link 2 [m].
    lc1 = p[9]    # Distance from ankle joint to CoM of link 1 [m].
    lc2 = p[10]   # Distance from knee joint to CoM of link 2 [m].
    lc3 = p[11]   # Distance from hip joint to CoM of link 3 [m].

    # Constant terms.
    k0 = 1/(m1+m2+m3)
    k1 = lc1*m1+l1*m2+l1*m3
    k2 = lc2*m2+l2*m3
    k3 = lc3*m3

    # Angular positions of the links.
    th1 = x[0]
    th2 = x[1]
    th3 = x[2]

    # Angular velocities of the links.
    om1 = x[3]
    om2 = x[4]
    om3 = x[5]

    # Sine and cosine functions.
    s1 = np.sin(th1)
    s12 = np.sin(th1+th2)
    s123 = np.sin(th1+th2+th3)
    c1 = np.cos(th1)
    c12 = np.cos(th1+th2)
    c123 = np.cos(th1+th2+th3)

    # Position coordinates of the CoM.
    xCoM = k0*(k1*c1+k2*c12+k3*c123)
    yCoM = k0*(k1*s1+k2*s12+k3*s123)

    # Velocity of the CoM.
    vxCoM = -om1*yCoM-om2*k0*(k2*s12+k3*s123)-om3*k0*k3*s123
    vyCoM = om1*xCoM+om2*k0*(k2*c12+k3*c123)+om3*k0*k3*c123

    # Measurement performed by the user of the PLLO.
    y = np.array([th3, xCoM, yCoM, om3, vxCoM, vyCoM])

    return y

# Compile the function below in Numba without involving the Python interpreter.
@jitwrapper(nopython=True)
def ThreeLinkRobotLQRHips(x, uh, p, xnom, unom, Kt):
    """"
    Dynamics of the three-link robot. The torque at the hips is applied by the 
    actuators of the PLLO under the authority of the finite-time horizon LQR
    controller. The torque, and forces at the shoulders are applied by the 
    user of the PLLO.

    Arguments

    x: State of the three-link robot.
      x[0]: angular position of link 1 relative to the horizontal in [rad].
      x[1]: angular position of link 2 relative to link 1 in [rad].
      x[2]: angular position of link 3 relative to link 2 in [rad].
      x[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      x[4]: angular velocity of link 2 in [rad/s].
      x[5]: angular velocity of link 3 in [rad/s].
    
    uh: Input from the user of the PLLO.
      uh[0]: torque applied by the user to link 3 at shoulder joints in [N.m].
      uh[1]: horizontal force applied by the user at shoulder joints in [N].
      uh[2]: vertical force applied by the user at shoulder joints in [N].
    
    p: Values for the parameters of the system.
      p[0]: Mass of link 1 in [kg].
      p[1]: Mass of link 2 in [kg].
      p[2]: Mass of link 3 in [kg].
      p[3]: Moment of inertia of link 1 about its Center of Mass (CoM) in [kg.m^2].
      p[4]: Moment of inertia of link 2 about its CoM in [kg.m^2].
      p[5]: Moment of inertia of link 3 about its CoM in [kg.m^2].
      p[6]: Length of link 1 in [m].
      p[7]: Length of link 2 in [m].
      p[8]: Length of link 3 in [m].
      p[9]: Distance from ankle joint to CoM of link 1 in [m].
      p[10]: Distance from knee joint to CoM of link 2 in [m].
      p[11]: Distance from hip joint to CoM of link 3 in [m].
      
    xnom: Reference values for computing state deviation variable dx.
      xnom[0]: angular position of link 1 relative to the horizontal in [rad].
      xnom[1]: angular position of link 2 relative to link 1 in [rad].
      xnom[2]: angular position of link 3 relative to link 2 in [rad].
      xnom[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      xnom[4]: angular velocity of link 2 in [rad/s].
      xnom[5]: angular velocity of link 3 in [rad/s].
      
    unom: Reference values obtained from the control allocation.
      unom[0]: torque at hip joints in [N.m].
      unom[1]: torque at shoulder joints in [N.m].
      unom[2]: horizontal force at shoulder joints in [N].
      unom[3]: vertical force at shoulder joints in [N].
    
    Output

    xdot: Time derivative of the state.
      xdot[0]: angular velocity of link 1 in [rad/s].
      xdot[1]: angular velocity of link 2 in [rad/s].
      xdot[2]: angular velocity of link 3 in [rad/s].
      xdot[3]: angular acceleration of link 1 in [rad/s^2].
      xdot[4]: angular acceleration of link 2 in [rad/s^2].
      xdot[5]: angular acceleration of link 3 in [rad/s^2].
     
    """
    
    # Aliases for numpy functions. 
    sin = np.sin
    cos = np.cos

    # State of the system.
    th = x[0:3]  # Angular positions.
    om = x[3:6]  # Angular velocities.

    # Terms with squares of angular velocities.
    omsq = np.array([om[0], om[0]+om[1], om[0]+om[1]+om[2]])**2

    # Trigonometric functions
    s2 = sin(th[1])
    s3 = sin(th[2])
    s23 = sin(th[1]+th[2])
    c2 = cos(th[1])
    c3 = cos(th[2])
    c23 = cos(th[1]+th[2])
    sv = sin(np.array([th[0], th[0]+th[1], th[0]+th[1]+th[2]]))
    cv = cos(np.array([th[0], th[0]+th[1], th[0]+th[1]+th[2]]))

    # Acceleration of gravity in [m/s^2].
    g = 9.81

    # Parameters of the system.
    m1 = p[0]     # Mass of link 1 [kg].
    m2 = p[1]     # Mass of link 2 [kg].
    m3 = p[2]     # Mass of link 2 [kg].
    I1 = p[3]     # Moment of inertia of link 1 about its CoM.
    I2 = p[4]     # Moment of inertia of link 2 about its CoM.
    I3 = p[5]     # Moment of inertia of link 3 about its CoM.
    l1 = p[6]     # Length of link 1 [m].
    l2 = p[7]     # Length of link 2 [m].
    l3 = p[8]     # Length of link 3 [m].
    lc1 = p[9]    # Distance from ankle joint to CoM of link 1 [m].
    lc2 = p[10]   # Distance from knee joint to CoM of link 2 [m].
    lc3 = p[11]   # Distance from hip joint to CoM of link 3 [m].

    # Constant terms.
    k1 = lc1*m1 + l1*m2 + l1*m3
    k2 = lc2*m2 + l2*m3
    k3 = lc3*m3

    # Mass matrix.
    M11 = I1 + I2 + I3 + lc1**2*m1 + m2*(l1**2 + 2*l1*lc2*c2 + lc2**2) + m3*(l1**2 + 2*l1*l2*c2 + 2*l1*lc3*c23 + l2**2 + 2*l2*lc3*c3 + lc3**2)
    M12 = I2 + I3 + lc2*m2*(l1*c2 + lc2) + m3*(l1*l2*c2 + l1*lc3*c23 + l2**2 + 2*l2*lc3*c3 + lc3**2)
    M13 = I3 + lc3*m3*(l1*c23 + l2*c3 + lc3)
    M22 = I2 + I3 + lc2**2*m2 + m3*(l2**2 + 2*l2*lc3*c3 + lc3**2)
    M23 = I3 + lc3*m3*(l2*c3 + lc3)
    M33 = I3 + lc3**2*m3
    M = np.array(((M11,M12,M13),
                  (M12,M22,M23),
                  (M13,M23,M33)))

    # Mass matrix inverse
    Minv = np.linalg.inv(M)

    # Vector of gravity effects.
    fg = -g*np.dot(
            np.array(((k1,k2,k3),
                      (0,k2,k3),
                      (0,0,k3))),
            cv)

    # Coriolis effects matrix.
    k4 = l1*(k2*s2+k3*s23)
    k5 = k3*l2*s3
    C = np.array(((k4,-k2*l1*s2+k3*l2*s3,-k3*l1*s23-k3*l2*s3),
                  (k4,k5,-k5),
                  (l1*k3*s23,k5,0)))

    # Gravity and Coriolis effects.
    f = fg-np.dot(C, omsq)

    # Matrix of generalized force.
    Atau = np.hstack((
        np.array([0,0,1]).reshape((3, 1)),
        -np.ones((3, 1)),
        -np.dot(np.array(((l1,l2,l3),
                   (0,l2,l3),
                   (0,0,l3))), sv).reshape((3, 1)),
        np.dot(np.array(((l1,l2,l3),
                  (0,l2,l3),
                  (0,0,l3))), cv).reshape((3, 1))))

    # State deviation variable.
    dx = x - xnom

    # Input from finite-time horizon LQR controller.
    du = -np.dot(Kt, dx)
    uctrl = unom + du

    # Merge inputs from the LQR controller, and the user of the PLLO.
    u = np.zeros((uctrl.shape[0],))
    u[0] = uctrl[0]
    u[1:uctrl.shape[0]] = uh

    # Time derivative of the state.
    xdot = np.zeros((x.shape[0],))
    xdot[0:3] = om
    xdot[3:6] = np.dot(Minv, f+ np.dot(Atau, u))

    return xdot

def ThreeLinkRobotDynamics(t, x, uh, p, fx, fu, fK):
    """
    Computes the dynamics of the three-link robot model due to the input 
    applied by the user of the PLLO at the shoulder joints, and the torque 
    commanded to the actuators of the hips by the onboard computer. The input 
    from the controller is determined by interpolation in time of the reference
    trajectories for the state and input, and the finite-time horizon LQR gain 
    that are designed for performing the STS movement.
    
    Arguments
    
    t: Time in [s].
    
    x: State of the three-link robot.
      x[0]: angular position of link 1 relative to the horizontal in [rad].
      x[1]: angular position of link 2 relative to link 1 in [rad].
      x[2]: angular position of link 3 relative to link 2 in [rad].
      x[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      x[4]: angular velocity of link 2 in [rad/s].
      x[5]: angular velocity of link 3 in [rad/s].
    
    uh: Input from the user of the PLLO.
      uh[0]: torque applied by the user to link 3 at shoulder joints in [N.m].
      uh[1]: horizontal force applied by the user at shoulder joints in [N].
      uh[2]: vertical force applied by the user at shoulder joints in [N].
    
    p: Values for the parameters of the system.
      p[0]: Mass of link 1 in [kg].
      p[1]: Mass of link 2 in [kg].
      p[2]: Mass of link 3 in [kg].
      p[3]: Moment of inertia of link 1 about its Center of Mass (CoM) in [kg.m^2].
      p[4]: Moment of inertia of link 2 about its CoM in [kg.m^2].
      p[5]: Moment of inertia of link 3 about its CoM in [kg.m^2].
      p[6]: Length of link 1 in [m].
      p[7]: Length of link 2 in [m].
      p[8]: Length of link 3 in [m].
      p[9]: Distance from ankle joint to CoM of link 1 in [m].
      p[10]: Distance from knee joint to CoM of link 2 in [m].
      p[11]: Distance from hip joint to CoM of link 3 in [m].
    
    fx: function in time for interpolation of reference state.
        
    fu: function in time for interpolation of reference input.
    
    fK: function in time for interpolation of finite-time horizon LQR gain. 
    
    Output
    
    xdot: Time derivative of the state.
      xdot[0]: angular velocity of link 1 in [rad/s].
      xdot[1]: angular velocity of link 2 in [rad/s].
      xdot[2]: angular velocity of link 3 in [rad/s].
      xdot[3]: angular acceleration of link 1 in [rad/s^2].
      xdot[4]: angular acceleration of link 2 in [rad/s^2].
      xdot[5]: angular acceleration of link 3 in [rad/s^2].    
    
    """
    
    # Interpolation in time of reference state.
    xnom = fx(t)

    # Interpolation in time of reference input.
    unom = fu(t)

    # Interpolation in time of finite-time horizon LQR gain.
    Kt = fK(t)
    
    # Compute dynamics of the three-link planar robot.
    xdot = ThreeLinkRobotLQRHips(x, uh, p, xnom, unom, Kt)

    return xdot


def LQRInput(t, x, fx, fu, fK):
    """
    Function for computing the input commanded by the finite-time horizon LQR 
    controller at time t.  
    
    Arguments
    
    t: Time in [s].
    
    x: State of the three-link robot. 
      x[0]: angular position of link 1 relative to the horizontal in [rad].
      x[1]: angular position of link 2 relative to link 1 in [rad].
      x[2]: angular position of link 3 relative to link 2 in [rad].
      x[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      x[4]: angular velocity of link 2 in [rad/s].
      x[5]: angular velocity of link 3 in [rad/s].
      
    fx: Function in time for interpolation of reference state.
        
    fu: Function in time for interpolation of reference input.
    
    fK: Function in time for interpolation of finite-time horizon LQR gain. 
    
    Output
    
    uctrl: Input commanded by the finite-time horizon LQR controller.
      u[0]: torque at hip joints in [N.m].
      u[1]: torque at shoulder joints in [N.m].
      u[2]: horizontal force at shoulder joints in [N].
      u[3]: vertical force at shoulder joints in [N].

    """
    # Interpolation in time of reference state.
    xnom = fx(t)

    # Interpolation in time of reference input.
    unom = fu(t)

    # Interpolation in time of finite-time horizon LQR gain.
    Kt = fK(t)

    # State deviation variable.
    dx = x - xnom

    # State feedback with finite-horizon LQR gain.
    du = -Kt.dot(dx)

    # Input commanded by the controller.
    u = unom + du

    return u

def ReferenceInterpolants(tgrid, xbar, ubar, ybar):
    """
    Build function handles for interpolating the reference trajectories of the 
    state, input, and the output measured by the user in the time domain.
    
    Arguments
    
    tgrid: n by 1 time array.  
        
    xbar: n by 6 reference state array.
      xbar[:,0]: angular position of link 1 relative to the horizontal in [rad].
      xbar[:,1]: angular position of link 2 relative to link 1 in [rad].
      xbar[:,2]: angular position of link 3 relative to link 2 in [rad].
      xbar[:,3]: angular velocity of link 1 in the inertial frame in [rad/s].
      xbar[:,4]: angular velocity of link 2 in [rad/s].
      xbar[:,5]: angular velocity of link 3 in [rad/s].
        
    ubar: n by 4 reference input array.
      ubar[:,0]: torque at hip joints in [N.m].
      ubar[:,1]: torque at shoulder joints in [N.m].
      ubar[:,2]: horizontal force at shoulder joints in [N].
      ubar[:,3]: vertical force at shoulder joints in [N].
        
    ybar: n by 6 reference output array.
      y[:,0]: angular position of link 3 relative to link 2 in [rad].
      y[:,1]: x coordinate of the CoM in [m].
      y[:,2]: y coordinate of the CoM in [m].
      y[:,3]: angular velocity of link 3 relative to link 2 [rad/s].
      y[:,4]: x velocity of the CoM in [m/s].
      y[:,5]: y velocity of the CoM in [m/s].

    Output
    
    fx: handle for interpolating the reference state as a funcion of time.
    
    fu: handle for interpolating the reference input as a funcion of time.
    
    fy: handle for interpolating the reference output measured by the user as 
    a funcion of time.

    """
    
    # Import data interpolators. 
    from scipy.interpolate import CubicSpline

    # Reshape time data.
    tgrid = tgrid.reshape((tgrid.shape[1],))
    
    # Interpolate time data with piecewise cubic polynomials.
    fx = CubicSpline(tgrid, xbar, axis=0)
    fu = CubicSpline(tgrid, ubar, axis=0)
    fy = CubicSpline(tgrid, ybar, axis=0)

    return fx, fu, fy

def InitialILCInput(t, T, u0, uT):
    """
    Conduct linear interpolation in the time domain for the torque and force 
    applied at the shoulder joints by the user of the PLLO, in order to 
    initialize the ILC algorithm.

    Arguments
    
    t: Time in [s]. 

    T: Duration of ascension phase in [s].
        
    u0: Input applied by the PLLO user at the beginning of the ascension phase.
      u0[0]: torque at shoulder joints in [N.m].
      u0[1]: horizontal force at shoulder joints in [N].
      u0[2]: vertical force at shoulder joints in [N].
      
    uT: Input applied by the PLLO user at the end of the ascension phase.
      uT[0]: torque at shoulder joints in [N.m].
      uT[1]: horizontal force at shoulder joints in [N].
      uT[2]: vertical force at shoulder joints in [N].

    Output
    
    uh: Linear interpolation of the input applied by the PLLO user at time t.
      uh[0]: torque at shoulder joints in [N.m].
      uh[1]: horizontal force at shoulder joints in [N].
      uh[2]: vertical force at shoulder joints in [N].
      
    """
    
    # Parameters for linear interpolation. 
    ml = (uT-u0)/T
    bl = uT-ml*T
        
    # Linear interpolation of the torque and forces at the shoulder joints.
    uh = ml*t + bl

    return uh

def LQRGainInterpolant(tgrid, K):
    """
    Conduct linear interpolation in the time domain for the finite-time 
    horizon LQR gain.  

    Arguments
    
    tgrid: n by 1 time array. 

    K: 4 by 6 by n array. 

    Output
    
    fK: handle for interpolating the finite-time horizon LQR matrix gain. 
      
    """
    
    # Import function for linear interpolation. 
    from scipy.interpolate import interp1d

    tgrid = tgrid.reshape((tgrid.shape[1],))
    fK = interp1d(tgrid, K, axis=2, fill_value="extrapolate")

    return fK

def ILCReward(t,y,fy,up,uc,dt,w):
    """
    Compute the reward for an iteration of ILC at time t.
    
    Arguments
    
    t: Time in [s].
    
    y: Output measured by the user of the PLLO at time t.
      x[0]: angular position of link 1 relative to the horizontal in [rad].
      x[1]: angular position of link 2 relative to link 1 in [rad].
      x[2]: angular position of link 3 relative to link 2 in [rad].
      x[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      x[4]: angular velocity of link 2 in [rad/s].
      x[5]: angular velocity of link 3 in [rad/s].
      
    fy: Function handle for interpolation of reference output.
        
    up: Input applied by the PLLO user at previous time sample.
      up[0]: torque at shoulder joints in [N.m].
      up[1]: horizontal force at shoulder joints in [N].
      up[2]: vertical force at shoulder joints in [N].
    
    uc: Input applied by the PLLO user at current time sample. 
      uc[0]: torque at shoulder joints in [N.m].
      uc[1]: horizontal force at shoulder joints in [N].
      uc[2]: vertical force at shoulder joints in [N].
      
    dt: Sampling time in [s]. 
    
    w: Weight to account for the different units of the output and the rate of 
    change of the input applied by the user of the PLLO.
    
    Output
    
    rew: Reward for an iteration of ILC at time t.

    """
    
    # Deviation from output reference trajectory.
    dy = fy(t) - y
    
    # Approximate time derivative of input.
    dudt = (uc-up)/dt
    
    # 2-norm reward from deviation of output reference trajectory.
    rew = -np.linalg.norm(dy) -w*np.linalg.norm(dudt)
    
    return rew


class STSEnv(gym.Env):
    """
    
    Environment object of the three-link planar robot modeling the STS movement. 

    The state of the system is a vector in R^6 where:
      x[0]: angular position of link 1 relative to the horizontal in [rad].
      x[1]: angular position of link 2 relative to link 1 in [rad].
      x[2]: angular position of link 3 relative to link 2 in [rad].
      x[3]: angular velocity of link 1 in the inertial frame in [rad/s].
      x[4]: angular velocity of link 2 in [rad/s].
      x[5]: angular velocity of link 3 in [rad/s].

    """

    metadata = {'render.modes': ['human'],}

    # Bounds for the input applied by the user at the shoulder joints.
    INPUT_MIN = np.zeros((3,))
    INPUT_MIN[0] = -175  # Minimum torque at the shoulder in [N.m].
    INPUT_MIN[1] = -40   # Minimum horizontal force at the shoulder in [N].
    INPUT_MIN[2] = 0     # Minimum vertical force at the shoulder in [N].
    
    INPUT_MAX = np.zeros((3,))
    INPUT_MAX[0] = 50    # Maximum torque at the shoulder in [N.m].
    INPUT_MAX[1] = 40    # Maximum horizontal force at the shoulder in [N].
    INPUT_MAX[2] = 650   # Maximum vertical force at the shoulder in [N].

    # Bounds for the output measured by the user of the PLLO.
    OBS_MIN = np.zeros((7,))
    OBS_MIN[0] = 0     # Minimum angular position of the torso relative to the thighs in [rad].
    OBS_MIN[1] = -0.5  # Minimum x position of the CoM in [m].
    OBS_MIN[2] = 0.5   # Minimum y position of the CoM in [m].
    OBS_MIN[3] = -80*np.pi/180 # Minimum angular velocity of the torso relative to the thighs in [rad/s].
    OBS_MIN[4] = -0.15 # Minimum x velocity of the CoM in [m/s].
    OBS_MIN[5] = 0     # Minimum y velocity of the CoM in [m/s].
    OBS_MIN[6] = 0

    OBS_MAX = np.zeros((7,))
    OBS_MAX[0] = 130*np.pi/180 # Maximum angular position of the torso relative to the thighs [rad].
    OBS_MAX[1] = 0.5   # Maximum x position of the Center of Mass in [m].
    OBS_MAX[2] = 1.025 # Maximum y position of the Center of Mass in [m].
    OBS_MAX[3] = 30*np.pi/180  # Maximum angular velocity of the torso relative to the thighs in [rad/s].
    OBS_MAX[4] = 0.02  # Maximum x velocity of the Center of Mass in [m/s].
    OBS_MAX[5] = 0.18  # Maximum y velocity of the Center of Mass in [m/s].
    OBS_MAX[6] = 3.5

    # Bounds for the state of the PLLO.
    STATE_MIN = np.zeros((6,))
    STATE_MIN[0] = 80*np.pi/180   # Minimum angular position of the shanks relative to the horizontal [rad].
    STATE_MIN[1] = -120*np.pi/180 # Minimum angular position of the thighs relative to the shanks [rad].
    STATE_MIN[2] = 0              # Minimum angular position of the torso relative to the thighs [rad]. 
    STATE_MIN[3] = -20*np.pi/180  # Minimum angular velocity of the shanks [rad/s].
    STATE_MIN[4] = -5*np.pi/180   # Minimum angular velocity of the thighs [rad/s].
    STATE_MIN[5] = -70*np.pi/180  # Minimum angular velocity of the torso [rad/s].

    STATE_MAX = np.zeros((6,))
    STATE_MAX[0] = 120*np.pi/180  # Maximum angular position of the shanks relative to the horizontal [rad].
    STATE_MAX[1] = 0              # Maximum angular position of the thighs relative to the shanks [rad].
    STATE_MAX[2] = 130*np.pi/180  # Maximum angular position of the torso relative to the thighs [rad].
    STATE_MAX[3] = 10*np.pi/180   # Maximum angular velocity of the shanks [rad/s].
    STATE_MAX[4] = 60*np.pi/180   # Maximum angular velocity of the thighs [rad/s].
    STATE_MAX[5] = 20*np.pi/180   # Maximum angular velocity of the torso [rad/s].
    
    def __init__(self, punc, verbose=True):

        # Sitting position on the z-space for STS 1.
        self._zi = np.array([-90*np.pi/180, 0.3099, 0.6678])

        # Standing position on the z-space.
        self._zf = np.array([-5*np.pi/180, 0, 0.9735])

        # Nominal values for the parameters of the system.
        self._pnom = np.array([9.68, 12.59, 44.57, 1.16456, 0.518821, 2.55731, 0.533, 0.406, 0.52, 0.533/2, 0.406/2, 0.52/2])

        # Values for the parameters of the system.
        self._punc = punc;

        # Sampling time.
        self._dt = 0.004 # in [s].

        # Total duration of ascension phase of the STS movement.
        self._tf = 3.5 # in [s].

        # Initial state of the system.
        self._x0 = np.array([90*np.pi/180, -90*np.pi/180, 90*np.pi/180, 0, 0, 0])

        # Import data for performing the STS movement from STS1dataTCST.mat file.
        data = loadmat('STS1dataTCST.mat')

        # Time array for interpolation of reference trajectories.
        self._tgrid = data["tgrid"]

        # Reference state trajectories.
        self._xbar = data["xbar1"]

        # Reference input trajectories.
        self._ubar = data["ubar1"]

        # Time varying LQR matrix gain.
        self._KLQR = data["KLQRSTS1"]

        # Time array for interpolation of time varying LQR matrix gain.
        self._tLQR = data["tgrid"]
        
        # Compute reference output trajectories. 
        self._ybar = np.empty((6,self._tgrid.shape[1]))
        for i in range(self._tgrid.shape[1]):
            self._ybar[:,i] = StateToOutput(self._pnom, self._xbar[i,:])
        self._ybar = self._ybar.transpose()

        # Obtain function handles to interpolate reference trajectories.
        self._fx, self._fu, self._fy = ReferenceInterpolants(self._tgrid, self._xbar, self._ubar, self._ybar)

        # Obtain function handle to interpolate the finite-time horizon LQR gain.
        self._fK = LQRGainInterpolant(self._tLQR, self._KLQR)
        
        # Weight to account for different units in the computation of the cost of the ILC iteration.
        self._wILC = 0.0001

        # Initial value of past input.
        self._up = self._fu(0)[1:]

        self.viewer = None

        self._verbose = verbose

        self.reset()

        # Bounds for input and output.
        self.action_space = gym.spaces.Box(low=self.INPUT_MIN, high=self.INPUT_MAX)
        self.observation_space = gym.spaces.Box(low=self.OBS_MIN, high=self.OBS_MAX)

    def step(self, u):
        """
        Function for computing the reward of the ILC algorithm and integrate
        the ODEs of the PLLO system.

        """
        
        # Check size of the input array. 
        assert u.shape == (3,)
        
        # Current observation.
        y = StateToOutput(self._punc, self._xcur)
        
        # Compute reward before transitioning the system.
        reward = ILCReward(self._tcur, y, self._fy, self._up, u, self._dt, self._wILC)

        if self._tcur >= self._tf:
            # Stop if the final integration time has been reached.
            if self._verbose:
                print("Final time for integration reached.")
            return self._xcur, self._get_obs(), reward, True
        else:
            # Integrate the ODEs forward by the sampling time.
            self._up = u
            self._ode.set_f_params(u)
            self._xcur = self._ode.integrate(self._tcur + self._dt)

            # Check state bounds at each integration step.
            x_max_exceeded = any(self._xcur > self.STATE_MAX)
            x_min_exceeded = any(self._xcur < self.STATE_MIN)
            
            if x_max_exceeded or x_min_exceeded:
                # Stop integration. 
                if self._verbose:
                    if x_max_exceeded:
                        print("Upper bound for the state reached at \nx=",self._xcur,".")
                    if x_min_exceeded:
                        print("Lower bound for the state reached at \nx=",self._xcur,".")
                reward = -1000.0
                return self._xcur, self._get_obs(), reward, True
            else:
                # Transition the system forward in time. 
                self._tcur += self._dt
                return self._xcur, self._get_obs(), reward, False

    def _ode_fn(self, t, y, u):
        # Dynamic equations of the PLLO system.
        return ThreeLinkRobotDynamics(t, y, u, self._punc, self._fx, self._fu, self._fK)

    def reset(self):
        # Define integration parameters.
        if self._verbose:
            print("Numerically solving the ODEs describing the system...")
        self._xcur = np.array(self._x0)
        self._tcur = 0
        self._up = self._fu(0)[1:]
        self._ode = scipy.integrate.ode(self._ode_fn)
        self._ode.set_integrator('dopri5')
        self._ode.set_initial_value(self._x0, 0)
        return self._get_obs()

    def cleanup_render_objects(self):
        # Delete objects used for the animation of the three-link planar robot. 
        if self.viewer is not None:
            print("Deleting viewer objects...")
            self.viewer.close()
            self.viewer = None
            del self.link1_transform
            del self.link2_transform
            del self.link3_transform

    def render(self, mode='human'):
        # Routines for generating the animation of the three-link planar robot. 
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            class RelativeTransform(rendering.Transform):
                def __init__(self, before, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
                    rendering.Transform.__init__(self, translation, rotation, scale)
                    self.before = before

                def enable(self):
                    self.before.enable()
                    rendering.Transform.enable(self)

                def disable(self):
                    rendering.Transform.disable(self)
                    self.before.disable()

            def make_link_with_joint(length, width):
                link = rendering.make_capsule(length, width)
                joint = rendering.make_circle(.05)
                joint.set_color(0, 0, 0)
                link.gs.append(joint)
                return link
            
            # Length of the links. 
            l1 = self._punc[6]
            l2 = self._punc[7]
            l3 = self._punc[8]

            # Set the size of the window for showing the animation. 
            self.viewer = rendering.Viewer(600, 600)
            box_bound = 1.75
            self.viewer.set_bounds(-box_bound, box_bound, -box_bound, box_bound)

            # Define the object for rendering the link 1.
            link1 = make_link_with_joint(l1, .1)
            link1.set_color(.3, .3, .8)
            self.link1_transform = rendering.Transform()
            link1.add_attr(self.link1_transform)
            self.viewer.add_geom(link1)

            # Define the object for rendering the link 2.
            link2 = make_link_with_joint(l2, .1)
            link2.set_color(.3, .3, .9)
            self.link2_transform = RelativeTransform(before=self.link1_transform)
            link2.add_attr(self.link2_transform)
            self.viewer.add_geom(link2)

            # Define the object for rendering the link 3.
            link3 = make_link_with_joint(l3, .1)
            link3.set_color(.3, .3, 1.0)
            self.link3_transform = RelativeTransform(before=self.link2_transform)
            link3.add_attr(self.link3_transform)
            self.viewer.add_geom(link3)
            
            # Apply translation transformations to links 2, and 3. 
            self.link2_transform.set_translation(l1, 0)
            self.link3_transform.set_translation(l2, 0)

        # Apply rotation transformations to links 1, 2, and 3.
        th1 = self._xcur[0]
        th2 = self._xcur[1]
        th3 = self._xcur[2]
        self.link1_transform.set_rotation(th1)
        self.link2_transform.set_rotation(th2)
        self.link3_transform.set_rotation(th3)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        pass

    def _get_obs(self):
        # Measurement performed by the user of the PLLO. 
        obs = StateToOutput(self._punc, self._xcur)
        return np.append(obs, self._tcur)

class ILCAgent(object):
    """
    Agent for the implementation of the ILC algorithm to replace the finite-time horizon LQR shoulder control.
    """
    def __init__(self, env, K, L):
        
        # Load environment with dynamics of the system. 
        self.env = env
        
        # Output feedback gain.
        self.K = K
        
        # Feedforward gain. 
        self.L = L
        
        # Load input bounds.
        self.umin = env.INPUT_MIN 
        self.umax = env.INPUT_MAX       
        
        # Load handle for interpolation of the reference output trajectory.
        self.ynom = env._fy

    def act(self, upsilon, fuj, fej, T, gamma):
        
        # Current simulation time.
        t = upsilon[-1]
        
        # Check if simulation time is the same as in the enviroment.
        assert t == self.env._tcur
        
        # Determine the human input uh from the ILC algorithm, according to the time T where the previous ILC iteration stopped.
        if t>=0 and t<=T:
            # Human input from ILC algorithm with recalling matrix gamma, output feedback gain K, and feedforward gain L.
            uh = gamma.dot(fuj(t)) + self.K.dot(self.ynom(t) - upsilon[:-1]) + self.L.dot(fej(t))
        else:
            # Ignore feedforward term once the integration has been stopped.   
            uh = gamma.dot(fuj(t)) + self.K.dot(self.ynom(t) - upsilon[:-1])
            
        # Saturate human input to keep it within bounds.
        uh = np.minimum(self.umax, np.maximum(uh, self.umin))
        
        return uh


# Main function.
if __name__ == '__main__':
        
    # Import modules. 
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline
    from time import time
    
    # Boolean flag for the animation of the three-link planar robot over the ILC iterations. 
    Animate = True
    
    # Output feedback gain for the ILC algorithm obtained from a derivative-free optimization.
    KDFO = np.array([[-100.60110383,  -53.25883234,   71.59923152, -123.88230487, -208.47269812,   66.0212473 ],
                     [57.97514685,  -21.05833162,  -47.92471925,  -24.66682686, 166.92545999,  -31.91644411],
                     [28.86072733,   20.20306968,  139.97420168,   46.61908613, -29.75345972,  123.52196133]])
    
    # Feedforward gain for the ILC algorithm obtained from a derivative-free optimization.
    LDFO = np.array([[-0.63809287,    2.37603863,   46.31018072,   -2.51414651, -5.25466641,  -28.89487164],
                     [-2.94791141,  -29.34284351,  -25.24252925,    2.85082183, -3.55854683,   24.48195597],
                     [-73.08983839,  -58.25284164,  114.13698546,   19.81980238, 13.09838889,  126.93011825]])
    
    # Import STS 1 data from 'STS1dataTCST.mat' file.
    data = loadmat('STS1dataTCST.mat')

    # Time array for reference trajectories.
    tgrid = data["tgrid"]                
    
    # Reference state trajectories.
    xbar = data["xbar1"]

    # LQR gain.
    KLQR = data["KLQRSTS1"]

    # Nominal parameters. 
    pnom = np.array([9.68, 12.59, 44.57, 1.16456, 0.518821, 2.55731, 0.533, 0.406, 0.52, 0.533/2, 0.406/2, 0.52/2])

    # Reference output trajectories.
    ybar = np.empty((6,tgrid.shape[1]))
    for i in range(tgrid.shape[1]):
        ybar[:,i] = StateToOutput(pnom, xbar[i,:])
    ybar = ybar.transpose()
    
    # Reference input trajectories.
    ubar = data["ubar1"]
    
    # Function handle for interpolation of the LQR gain. 
    fK = LQRGainInterpolant(tgrid, KLQR)
    
    # Function handles to interpolate the reference trajectories.
    fxref, furef, fy = ReferenceInterpolants(tgrid, xbar, ubar, ybar)
    
    # Reshape time array.
    tgrid = tgrid.reshape((tgrid.shape[1],))
    
    # Sampling time for integration of the ODEs of the system.
    dt = 0.004
    
    # Number of integration steps per ILC iteration. 
    maxsteps = np.ceil(tgrid[-1]/dt + 1)
    maxsteps = maxsteps.astype(int)

    # Time array for simulated trajectories.
    tsim = np.linspace(tgrid[0],tgrid[-1],maxsteps)
     
    # Input applied by the user of the PLLO in the initial iteration of the ILC algorithm.
    U0 = np.empty((maxsteps,3))
    for t in range(maxsteps):
        U0[t,:] = InitialILCInput(t*dt, tgrid[-1], ubar[0,1:], ubar[-1,1:]) 
    
    # Enviroment of the three-link planar robot used for modeling the STS movement.
    env = STSEnv(pnom)
    
    # Initialize agent for the implementation of the ILC algorithm.
    agent = ILCAgent(env, KDFO, LDFO)
    
    # Number of ILC iterations. 
    ILC_count = 30
    
    # Array for rewards accross ILC iterations.  
    ILCRew = np.empty((ILC_count,))
    
    # Array for the cost of each ILC iteration.
    ILCcost = np.empty((ILC_count,))  
    
    # Array for keeping track of the number of ILC iterations.
    ILCiter = np.array(range(ILC_count)) + 1
    
    # Flag for stopping an ILC iteration. 
    done = False
    
    # Total duration of STS movement.
    T = tgrid[-1]
    
    # Function handle for interpolation of the initial input sequence applied by the user of the PLLO.
    fu = CubicSpline(tsim, U0, axis=0)
    
    # Function handle for interpolation of the deviation of the output measured by the user of the PLLO with respect to its reference trajectory.
    fe = CubicSpline(tgrid, np.zeros(ybar.shape), axis=0)
        
    # ILC algorithm.
    print("\nStarting ILC algorithm")
    ILCstart = time()
    for i in range(ILC_count):
        print("\nIteration", i+1)
        
        # Initialize integration time step index. 
        t = 0
        
        # Initial state.
        x = xbar[t,:]

        # Initial measurement by the user of the PLLO. 
        ob = env.reset()
        
        # Array for states over integration time steps. 
        xsim = np.empty((maxsteps,6))
        
        # Array for LQR input over integration time steps. 
        uLQR = np.empty((maxsteps,4))
        
        # Array for measurement performed by the PLLO user over integration time steps. 
        ysim = np.empty((maxsteps,6))
        
        # Array for output error over integration time steps. 
        ej = np.zeros((maxsteps,6)) 
        
        # Array for inputs over integration time steps. 
        usim = np.empty((maxsteps,3)) 
        
        # Recalling matrix. 
        rmat = np.identity(3)
        
        # Initialize rewards. 
        iter_reward = 0       # Accumulated reward for ILC iteration. 
        reward = 0            # Reward for integration time step. 
        
        # Array to track rewards for each integration time step in the current ILC iteration. 
        IterRew = np.empty((maxsteps+1,)) 
        IterRew[t] = reward 
        
        while True:
            # State.
            xsim[t,:] = x
            
            # LQR input.
            uLQR[t,:] = LQRInput(tsim[t], x, fxref, furef, fK)
            
            # Measurement performed by the user of the PLLO.
            ysim[t,:] = ob[:6]  
            
            # Deviation of the measurement performed by the user from the output reference trajectory.
            ej[t,:] = fy(t*dt) - ob[:6]
            
            # Input from ILC algorithm. 
            action = agent.act(ob, fu, fe, T, rmat)
            
            # Input applied by the user of the PLLO. 
            usim[t,:] = action
            
            # Integrate ODEs of the system subject to the input from the user of the PLLO. 
            x, ob, reward, done = env.step(action)
            
            # Advance to next integration time step. 
            t += 1
            
            # Store reward for the integration step. 
            IterRew[t] = reward 
            
            # Accumulate reward for ILC iteration.
            iter_reward += reward
            
            # Show animation of STS movement achieved in last ILC iteration.
            if Animate:
                if not (t % 5):
                    # Refresh image every 5 time steps. 
                    env.render()
            
            # Tasks that must be performed once an ILC iteration is stopped.
            if done:
                # Store reward achieved by the ILC iteration.
                ILCRew[i] = iter_reward
                # Compute input sequence for next iteration. 
                if t >= maxsteps:
                    uj = usim
                    T = tgrid[-1]
                else:
                    # Time before integration stopped.
                    T = (t-2)*dt
                    
                    # Input for next ILC iteration. 
                    uj = np.empty((maxsteps,3))
                    
                    # Parameters for linear extrapolation of the input in the next iteration. 
                    muext = (U0[-1,:] - usim[t-2,:])/(tgrid[-1]-T) 
                    buext = U0[-1,:] - muext*tgrid[-1]
                    
                    # Compute input for starting the next iteration.
                    for j in range(maxsteps):
                        if j <= t-2:
                            # Input sequence before integration stopped. 
                            uj[j,:] = usim[j,:]
                        else:
                            # Linear extrapolation of input used for the time horizon ahead of when the past integration stopped.
                            uj[j,:] = muext*(j*dt) + buext
                
                # Update interpolant of ILC input for next iteration.
                fu = CubicSpline(tsim, uj, axis=0)
                
                # Update interpolant of output error for next iteration.
                fe = CubicSpline(tsim, ej, axis=0)
                break   
        
        # Cost of ILC iteration.  
        ILCcost[i] = -iter_reward
        if ILCcost[i] >= 1000:
            print("STS movement stopped at t=", '%.3f'%T, "[s].\nThe iteration cost is infinity.")
        else:
            print("STS movement completed at t=", '%.3f'%T,"[s].\nThe iteration cost is", '%.2f'%ILCcost[i])
    
    # Total time for simulation of the ILC algorithm. 
    tElapsed = time() - ILCstart
    print("\nThe total time for performing", ILC_count,"iterations of the ILC algorithm was", '%.2f'%tElapsed,"[s].")
    
    # Clear objects for rendering the animation of the STS movement. 
    if Animate:
        input('Press Enter to clear the figure with the animation, and proceed to plot the results from the simulation.\n')
        env.cleanup_render_objects()

    # Plot state trajectories for the final iteration of the ILC algorithm.
    # Reference trajectories are denoted with hats. 
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    plt.figure()
    plt.suptitle('State trajectories of the three-link planar robot achieved after N=' + str(ILC_count) + ' iterations of the ILC algorithm', fontsize=18, fontweight='bold')
    plt.subplot(2,3,1)
    plt.plot(tgrid,xbar[:,0]*180/np.pi,'r--',tsim,xsim[:,0]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\theta_{1}(t)\:[deg]$', fontsize=18)
    plt.legend([r'$\hat{\theta}_1(t)$',r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best',fontsize=18)
    plt.subplot(2,3,2)
    plt.plot(tgrid,xbar[:,1]*180/np.pi,'r--',tsim,xsim[:,1]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\theta_{2}(t)\:[deg]$', fontsize=18)
    plt.legend([r'$\hat{\theta}_2(t)$',r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.subplot(2,3,3)
    plt.plot(tgrid,xbar[:,2]*180/np.pi,'r--',tsim,xsim[:,2]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\theta_{3}(t)\:[deg]$', fontsize=18)
    plt.legend([r'$\hat{\theta}_3(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.subplot(2,3,4)
    plt.plot(tgrid,xbar[:,3]*180/np.pi,'r--',tsim,xsim[:,3]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\dot{\theta}_{1}(t)\:[deg/s]$', fontsize=18)
    plt.legend([r'$\dot{\hat{\theta}}_1(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.subplot(2,3,5)
    plt.plot(tgrid,xbar[:,4]*180/np.pi,'r--',tsim,xsim[:,4]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\dot{\theta}_{2}(t)\:[deg/s]$', fontsize=18)
    plt.legend([r'$\dot{\hat{\theta}}_2(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.subplot(2,3,6)
    plt.plot(tgrid,xbar[:,5]*180/np.pi,'r--',tsim,xsim[:,5]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]',fontsize=18)
    plt.ylabel(r'$\dot{\theta}_{3}(t)\:[deg/s]$', fontsize=18)
    plt.legend([r'$\dot{\hat{\theta}}_3(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)

    # Plot trajectories of the output measured by the user of the PLLO in the last iteration of the ILC algorithm.
    # Reference trajectories are shown in dashed lines. 
    plt.figure()
    plt.suptitle('Trajectories of the output measured by the user of the PLLO after N=' + str(ILC_count) + ' iterations of the ILC algorithm', fontsize=18, fontweight='bold')
    plt.subplot(2,4,1)
    plt.plot(tgrid,ybar[:,0]*180/np.pi,'r--',tsim,ysim[:,0]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\theta_3(t)\:[deg]$', fontsize=18)
    plt.legend([r'$\hat{\theta}_3(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('Angular Position of Torso', fontsize=18)
    plt.subplot(2,4,2)
    plt.plot(tgrid,ybar[:,1],'r--',tsim,ysim[:,1],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$x_{CoM}(t)\:[m]$', fontsize=18)
    plt.legend([r'$\hat{x}_{CoM}(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('x coordinate of the CoM Position', fontsize=18)
    plt.subplot(2,4,3)
    plt.plot(tgrid,ybar[:,2],'r--',tsim,ysim[:,2],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$y_{CoM}(t)\:[m]$', fontsize=18)
    plt.legend([r'$\hat{y}_{CoM}(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('y Coordinate of the CoM Position', fontsize=18)
    plt.subplot(2,4,4)
    plt.plot(ysim[0,1],ysim[0,2],'ko', ysim[-1,1], ysim[-1,2], 'rX', ybar[:,1],ybar[:,2],'r--',ysim[:,1],ysim[:,2],'g', markersize=14)
    plt.grid()
    plt.xlabel(r'$x_{CoM} [m]$', fontsize=18)
    plt.ylabel(r'$y_{CoM} [m]$', fontsize=18)
    plt.legend(['Initial Reference Position', 'Final Reference Position', 'Reference Trajectory', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('CoM Position in the Sagittal Plane', fontsize=18)
    plt.subplot(2,4,5)
    plt.plot(tgrid,ybar[:,3]*180/np.pi,'r--',tsim,ysim[:,3]*180/np.pi,'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\dot{\theta}_3(t)\:[deg/s]$', fontsize=18)
    plt.legend([r'$\dot{\hat{\theta}}_3(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('Angular Velocity of Torso', fontsize=18)
    plt.subplot(2,4,6)
    plt.plot(tgrid,ybar[:,4],'r--',tsim,ysim[:,4],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\dot{x}_{CoM}(t)\:[m/s]$', fontsize=18)
    plt.legend([r'$\dot{\hat{x}}_{CoM}(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('x coordinate of the CoM Velocity', fontsize=18)
    plt.subplot(2,4,7)
    plt.plot(tgrid,ybar[:,5],'r--',tsim,ysim[:,5],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\dot{y}_{CoM}(t)\:[m/s]$', fontsize=18)
    plt.legend([r'$\dot{\hat{y}}_{CoM}(t)$', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('y Coordinate of the CoM Velocity', fontsize=18)
    plt.subplot(2,4,8)
    plt.plot(ysim[0,4], ysim[0,5],'ko', ysim[-1,4], ysim[-1,5], 'rX', ybar[:,4], ybar[:,5], 'r--', ysim[:,4], ysim[:,5], 'g', markersize=14)
    plt.grid()
    plt.xlabel(r'$\dot{x}_{CoM} [m/s]$', fontsize=18)
    plt.ylabel(r'$\dot{y}_{CoM} [m/s]$', fontsize=18)
    plt.legend(['Initial Reference Velocity','Final Reference Velocity','Reference Trajectory', r'For $\mu^{'+ str(ILC_count) +'}(t)$, $\gamma_{j}=I_3$, $p=\hat{p}$'], loc='best', fontsize=18)
    plt.title('CoM Velocity in the Sagittal Plane', fontsize=18)

    # Plot the loads applied at the joints of the three-link planar robot. 
    # Reference trajectories are shown in dashed lines.
    plt.figure()
    plt.suptitle('Loads applied at the joints of the three-link planar robot under the action of the ILC algorithm', fontsize=18, fontweight='bold')
    plt.subplot(2,2,1)
    plt.plot(tgrid,ubar[:,0],'r--',tsim,uLQR[:,0],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\tau_{h}(t)\:[N.m]$', fontsize=18)
    plt.legend([r'$\hat{\tau}_h(t)$', 'Input from LQR controller'], loc='best', fontsize=18)
    plt.title(r'Torque commanded at the hip joints by the LQR controller after $N=$' + str(ILC_count) + ' iterations of the ILC algorithm', fontsize=18)
    plt.subplot(2,2,2)
    plt.plot(tgrid,ubar[:,1],'r--',tsim,U0[:,0],'m',tsim,usim[:,0],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$\tau_{s}(t)\:[N.m]$', fontsize=18)
    plt.legend([r'$\hat{\tau}_s(t)$', r'$\mu_{1}^{0}(t)$',r'$\mu_{1}^{' + str(ILC_count) + '}(t)$'], loc='best', fontsize=18)
    plt.title(r'Torque applied at the shoulder joints by the ILC algorithm', fontsize=18)
    plt.subplot(2,2,3)
    plt.plot(tgrid,ubar[:,2],'r--',tsim,U0[:,1],'m',tsim,usim[:,1],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$F_x(t)\:[N]$', fontsize=18)
    plt.legend([r'$\hat{F}_x(t)$',r'$\mu_{2}^{0}(t)$',r'$\mu_{2}^{' + str(ILC_count) + '}(t)$'], loc='best', fontsize=18)
    plt.title(r'Horizontal force applied at the shoulder joints by the ILC algorithm', fontsize=18)
    plt.subplot(2,2,4)
    plt.plot(tgrid,ubar[:,3],'r--',tsim,U0[:,2],'m',tsim,usim[:,2],'g')
    plt.xlim(tgrid[0],tgrid[-1])
    plt.grid()
    plt.xlabel('t [s]', fontsize=18)
    plt.ylabel(r'$F_y(t)\:[N]$', fontsize=18)
    plt.legend([r'$\hat{F}_y(t)$',r'$\mu_{3}^{0}(t)$',r'$\mu_{3}^{' + str(ILC_count) + '}(t)$'], loc='best', fontsize=18)
    plt.title(r'Vertical force applied at the shoulder joints by the ILC algorithm', fontsize=18)
    plt.show()
    
    # Compute reference cost for the ILC algorithm.
    refcost = 0
    for i in np.arange(1,tsim.shape[0]-1):
        refcost += -ILCReward(tsim[i],fy(tsim[i]),fy,furef(tsim[i-1])[1:],furef(tsim[i])[1:],0.004,1e-4)
    
    # Plot cost per iteration of the ILC algorithm.
    plt.figure()
    infidx = np.where(ILCcost>=1000)
    infval = np.zeros([ILC_count,])
    finidx = np.where(ILCcost<1000)
    plt.plot(ILCiter[infidx], infval[infidx], 'bx', ILCiter[finidx], ILCcost[finidx], 'bo', np.array([ILCiter[0], ILCiter[-1]]), np.array([refcost,refcost]), 'r--')
    plt.grid()
    plt.xlabel('ILC Iteration j', fontsize=18)
    plt.ylabel('Cost per ILC iteration', fontsize=18)
    plt.legend([r'$J_{L}^{j}=\infty$', r'$J_{L}^{j}$',r'$\hat{J}_{L}$'], loc='best', fontsize=18)
    plt.suptitle(r'Cost $J_{L}^{j}$ attained by plugging $K^{\star}$ and $L^{\star}$ into the ILC algorithm with $\gamma_{j}=I_{3}$, and $p=\hat{p}$', fontsize=18, fontweight='bold')
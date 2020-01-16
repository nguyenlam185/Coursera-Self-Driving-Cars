import pickle
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init = data['x_init']  # initial x position [m]
y_init = data['y_init']  # initial y position [m]
th_init = data['th_init']  # initial theta position [rad]

# input signal
v = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.01  # translation velocity variance
om_var = 0.01  # rotational velocity variance
r_var = 0.1  # range measurements variance
b_var = 0.1  # bearing measurement variance

Q_km = np.diag([v_var, om_var])  # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init])  # initial state
P_est[0] = np.diag([1, 1, 0.1])  # initial state covariance

R = np.eye(2) * (10.0 ** 2)  # Tell the Kalman filter how bad the sensor readings are
x_check = np.zeros((3, 1))
F = np.zeros((3, 3))
P_check = np.eye(3)
I = np.eye(3)


# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x


def measurement_update(lk, rk, bk, P_check, x_check):
    # 1. Compute measurement Jacobian
    xk = x_check[0]
    yk = x_check[1]
    theta_k = x_check[2]
    xl = lk[0]
    yl = lk[1]

    H = np.zeros([2, 3])
    H[0, 0] = (d * np.cos(theta_k) + xk - xl) / np.sqrt(((((-d) * np.sin(theta_k)) - yk + yl) ** 2)
                                                        + ((((-d) * np.cos(theta_k)) - xk + xl) ** 2))
    H[0, 1] = (d * np.sin(theta_k) + yk - yl) / np.sqrt(((((-d) * np.sin(theta_k)) - yk + yl) ** 2)
                                                        + ((((-d) * np.cos(theta_k)) - xk + xl) ** 2))
    H[0, 2] = ((-d * (-d * np.sin(theta_k) - yk + yl) * np.cos(theta_k))
               + (-d * (-d * np.cos(theta_k) - xk + xl) * np.sin(theta_k))) \
              / np.sqrt(((((-d) * np.sin(theta_k)) - yk + yl) ** 2)
                        + ((((-d) * np.cos(theta_k)) - xk + xl) ** 2))
    H[1, 0] = (d * np.sin(theta_k) + yk - yl) / (((((-d) * np.sin(theta_k)) - yk + yl) ** 2)
                                                 + ((((-d) * np.cos(theta_k)) - xk + xl) ** 2))
    H[1, 1] = (d * np.cos(theta_k) + xk - xl) / (((((-d) * np.sin(theta_k)) - yk + yl) ** 2)
                                                 + ((((-d) * np.cos(theta_k)) - xk + xl) ** 2))
    H[1, 2] = (((d * (d * np.sin(theta_k) + yk - yl) * np.sin(theta_k))
                - (d * (((-d) * np.cos(theta_k)) - xk + xl) * np.cos(theta_k)))
               / (((((-d) * np.sin(theta_k)) - yk + yl) ** 2)
                  + ((((-d) * np.cos(theta_k)) - xk + xl) ** 2))) - 1

    M = np.eye(2)

    # 2. Compute Kalman Gain
    K = P_check @ H.T @ (inv(H @ P_check @ H.T + M @ R @ M.T))

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])

    # 4. Correct covariance

    x_check[2] = wraptopi(x_check[2])
    return x_check, P_check


#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    x_check = x_est[k-1, :] \
              + (delta_t * np.array([[np.cos(x_est[k-1, 2]), 0], [np.sin(x_est[k-1, 2]), 0], [0, 1]])
                 @ (np.array([[v[k]], om[k]]) + ))


    # 2. Motion model jacobian with respect to last state
    F_km = np.zeros([3, 3])
    F_km[0, 0] = 1
    F_km[0, 1] = 0
    F_km[0, 2] = (-delta_t) * v[k] * np.sin(x_check[2])
    F_km[1, 0] = 0
    F_km[1, 1] = 1
    F_km[1, 2] = delta_t * v[k] * np.cos(x_check[2])
    F_km[2, 0] = 0
    F_km[2, 1] = 0
    F_km[2, 2] = 1

    # 3. Motion model jacobian with respect to noise
    L_km = np.zeros([3, 2])

    # 4. Propagate uncertainty

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()

with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)

import pickle
from numpy import *
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
r_var = 0.01  # range measurements variance
b_var = 1  # bearing measurement variance

Q_km = diag([v_var, om_var])  # input noise covariance
cov_y = diag([r_var, b_var])  # measurement noise covariance

x_est = zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = array([x_init, y_init, th_init])  # initial state
P_est[0] = diag([1, 1, 0.1])  # initial state covariance

# R = eye(2) * (10.0 ** 2)  # Tell the Kalman filter how bad the sensor readings are
# x_check = zeros((3, 1))
# F = zeros((3, 3))
P_check = eye(3)
I = eye(3)

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > pi:
        x = x - (floor(x / (2 * pi)) + 1) * 2 * pi
    elif x < -pi:
        x = x + (floor(x / (-2 * pi)) + 1) * 2 * pi
    return x


def measurement_update(lk, rk, bk, P_check, x_check):
    """[summary]

    Args:
        lk ([type]): Ground truth x,y positions of landmarks
        rk ([type]): Ground truth range measurements
        bk ([type]): Ground truth bearing to each landmarks
        P_check ([type]): [description]
        x_check ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 1. Compute measurement Jacobian
    xk = x_check[0]
    yk = x_check[1]
    theta_k = wraptopi(x_check[2])
    xl = lk[0]
    yl = lk[1]

    d_x = xl - xk - d*cos(theta_k)
    d_y = yl - yk - d*sin(theta_k)
    r = sqrt(d_x**2 + d_y**2)

    H_k = zeros([2, 3])

    H_k[0,0] = -d_x/r
    H_k[0,1] = -d_y/r
    H_k[0,2] = d*(d_x*sin(theta_k) - d_y*cos(theta_k))/r
    H_k[1,0] = d_y/r**2
    H_k[1,1] = -d_x/r**2
    H_k[1,2] = -1-d*(d_y*sin(theta_k) + d_x*cos(theta_k))/r**2

    # 2. Compute Kalman Gain
    M_k = identity(2)
    K_k = P_check @ H_k.T @ (inv(H_k @ P_check @ H_k.T + M_k @ cov_y @ M_k.T))

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    y_l_k = vstack([rk, wraptopi(bk)])
    y_l_k_check = vstack([r, wraptopi(arctan2(d_y, d_x) - theta_k)])

    x_check = x_check + K_k @ (y_l_k - y_l_k_check)
    x_check[2] = wraptopi(x_check[2])

    # 4. Correct covariance
    P_check = (I - K_k @ H_k) @ P_check
    
    return x_check, P_check


#### 5. Main Filter Loop #######################################################################
# set the initial values
P_check = P_est[0]
x_check = x_est[0, :].reshape(3,1)
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    theta = wraptopi(x_check[2])

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
#     x_check = zeros(3)
    F = array([[cos(theta), 0],
               [sin(theta), 0],
               [0, 1]], dtype='float')
    inp = array([[v[k-1]], [om[k-1]]])

    x_check = x_check + (F @ inp) * delta_t
    x_check[2] = wraptopi(x_check[2])

    # 2. Motion model jacobian with respect to last state
    F_km = zeros([3, 3])
    F_km = array([[1, 0, -sin(theta)*delta_t*v[k-1]],
                     [0, 1, cos(theta)*delta_t*v[k-1]],
                     [0, 0, 1]], dtype='float')
    # dtype='float'
    # 3. Motion model jacobian with respect to noise
    L_km = zeros([3, 2])
    L_km = array([[cos(theta)*delta_t, 0], 
                    [sin(theta)*delta_t, 0],
                    [0,1]], dtype='float')

    # 4. Propagate uncertainty
    P_check = F_km @ P_check @ (F_km.T) + L_km @ Q_km @ L_km.T 

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Estimated trajectory')
ax1.plot(x_est[:, 0], x_est[:, 1])
ax2.plot(t[:], x_est[:, 2])
plt.show()

with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)

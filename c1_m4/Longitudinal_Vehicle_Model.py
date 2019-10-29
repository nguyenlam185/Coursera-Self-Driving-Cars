import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Vehicle():
    def __init__(self):
        # ==================================
        #  Parameters
        # ==================================

        # Throttle to engine torque
        self.a_0 = 400
        self.a_1 = 0.1
        self.a_2 = -0.0002

        # Gear ratio, effective radius, mass + inertia
        self.GR = 0.35
        self.r_e = 0.3
        self.J_e = 10
        self.m = 2000
        self.g = 9.81

        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01

        # Tire force
        self.c = 10000
        self.F_max = 10000

        # State variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

        self.sample_time = 0.01

    def reset(self):
        # reset state variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

    def step(self, throttle, alpha):
        # ==================================
        #  Implement vehicle model here
        # ==================================

        # Engine Torque
        t_e = throttle * (self.a_0 + (self.a_1 * self.w_e) + (self.a_2 * self.w_e * self.w_e))

        # Aerodynamic drag
        f_aero = self.c_a * self.v * self.v

        # Rx
        r_x = self.c_r1 * self.v

        # Fg
        f_g = self.m * self.g * np.sin(alpha)

        # Fload
        f_load = f_aero + r_x + f_g

        # w_w
        w_w = self.GR * self.w_e

        s = ((w_w * self.r_e) - self.v) / self.v

        if (abs(s) < 1):
            f_x = self.c
        else:
            f_x = self.F_max

        # x_dot_dot
        x_dot_dot = (f_x - f_load) / self.m

        self.a = x_dot_dot
        self.v += self.a
        self.x += self.v


sample_time = 0.01
model = Vehicle()
time_end = 20
t_data = np.arange(0, time_end, sample_time)
x_data = np.zeros_like(t_data)

# reset the states
model.reset()


# ==================================
#  Learner solution begins here
# ==================================
def slope_intercept(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


throttle_data = np.zeros_like(t_data)
alpha_data = np.zeros_like(t_data)

a1, b1 = slope_intercept(0, 0.2, 5, 0.5)
for i in range(500):
    throttle_data[i] = a1 * t_data[i] + b1
    alpha_data[i] = 0.05  # = atan(3/60)

for i in range(500, 1500, 1):
    throttle_data[i] = 0.5
    alpha_data[i] = 0.1  # = atan(3/30)

a2, b2 = slope_intercept(15, 0.5, 20, 0)
for i in range(1500, 2000, 1):
    throttle_data[i] = a2 * t_data[i] + b2

for i in range(t_data.shape[0]):
    x_data[i] = model.v
    model.step(throttle_data[i], alpha_data[i])
# ==================================
#  Learner solution ends here
# ==================================

# Plot x vs t for visualization
plt.subplot(2, 1, 1)
plt.plot(t_data, throttle_data)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t_data, x_data)
plt.grid()
plt.show()
print('Good bye')

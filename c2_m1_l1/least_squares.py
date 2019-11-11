import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# First we will import the neccesary Python modules and load the current and voltage measurements into numpy arrays:
# Store the voltage and current data as column vectors.
I = np.mat([0.2, 0.3, 0.4, 0.5, 0.6]).T
V = np.mat([1.23, 1.38, 2.06, 2.47, 3.17]).T

# Now we can plot the measurements - can you see the linear relationship between current and voltage?
plt.scatter(np.asarray(I), np.asarray(V))

plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()

# Define the H matrix, what does it contain?
H = np.mat([1,1,1,1,1]).T

# Now estimate the resistance parameter.
R = np.dot(np.dot(inv(np.dot(H.T, H)), H.T), (V/I))
print('The slope parameter (i.e., resistance) for the best-fit line is:')
print(R)

# Now let's plot our result. How do we relate our linear parameter fit to the resistance value in ohms?

I_line = np.arange(0, 0.8, 0.1)
V_line = R*I_line

plt.scatter(np.asarray(I), np.asarray(V))
plt.plot(I_line, np.asarray(V_line).reshape(-1))
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()
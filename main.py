# https://www.bilibili.com/video/BV1Lg411p7mN/?spm_id_from=333.999.0.0&vd_source=cc9173a26c5dd3ff48bdde7ba3fb64f2
# 最优控制第3讲动态规划问题Python代码版本

import numpy as np
import matplotlib.pyplot as plt

# Define IC (Initial Conditions)
h_init = 0  # h: height
v_init = 0  # v: velocity

# Final state
h_final = 10
v_final = 0

# Boundary condition
h_min = 0
h_max = 10
N_h = 10

v_min = 0
v_max = 3
N_v = 50

# Create state arrays (discretizing height and velocity)
Hd = np.linspace(h_min, h_max, N_h + 1)  # Height states
Vd = np.linspace(v_min, v_max, N_v + 1)  # Velocity states

# Input constraint (system acceleration)
u_min = -3
u_max = 2

# Define cost to go matrix
J_costtogo = np.zeros((N_h + 1, N_v + 1))

# Define input acceleration matrix
Input_acc = np.zeros((N_h + 1, N_v + 1))

# Calculate average speed (v_avg)
v_avg = 0.5 * (v_final + Vd)

# Calculate travel time (T_delta), which is the cost
# Avoid division by zero if v_avg is zero
T_delta = np.where(v_avg == 0, np.inf, (h_max - h_min) / (N_h * v_avg))

# Calculate acceleration (acc)
acc = (v_final - Vd) / T_delta

# Assign delta T to cost to go (J_temp)
J_temp = T_delta.copy()

# Find which acc is over the limit
Ind_lin_acc = np.where((acc < u_min) | (acc > u_max))

# Let certain elements be infinity in J_temp
J_temp[Ind_lin_acc] = np.inf

# Save to cost to go matrix
J_costtogo[1, :] = J_temp

# Save to acc matrix
Input_acc[1, :] = acc

# --- From 8m to 2m ---
# Prepare the matrix
Vd_x, Vd_y = np.meshgrid(Vd, Vd)

# Calculate average speed matrix (v_avg)
v_avg = 0.5 * (Vd_x + Vd_y)

# Calculate travel time (T_delta), which is the cost
# Avoid division by zero
T_delta = np.where(v_avg == 0, np.inf, (h_max - h_min) / (N_h * v_avg))

# Calculate acceleration (acc)
acc = (Vd_y - Vd_x) / T_delta

for k in range(2, N_h):
    # Assign delta T to cost to go (J_temp)
    J_temp = T_delta.copy()

    # Find which acc is over the limit
    Ind_lin_acc = np.where((acc < u_min) | (acc > u_max))

    # Let certain elements be infinity in J_temp
    J_temp[Ind_lin_acc] = np.inf

    # Add cost to go from the previous step
    for i in range(J_temp.shape[1]):
        J_temp[:, i] += J_costtogo[k - 1, :]

    # Save to cost to go matrix
    J_costtogo[k, :] = np.min(J_temp, axis=0)
    l = np.argmin(J_temp, axis=0)

    # Save to acc matrix
    Ind_lin_acc = np.ravel_multi_index((l, np.arange(len(l))), J_temp.shape)
    Input_acc[k, :] = acc.ravel()[Ind_lin_acc]

# --- From 2m to 0m ---
# Calculate average speed matrix (v_avg)
v_avg = 0.5 * (Vd + v_init)

# Calculate travel time (T_delta), which is the cost
T_delta = np.where(v_avg == 0, np.inf, (h_max - h_min) / (N_h * v_avg))

# Calculate acceleration (acc)
acc = (Vd - v_init) / T_delta

# Assign delta T to cost to go (J_temp)
J_temp = T_delta.copy()

# Find which acc is over the limit
Ind_lin_acc = np.where((acc < u_min) | (acc > u_max))

# Let certain elements be infinity in J_temp
J_temp[Ind_lin_acc] = np.inf

# Add cost to go from the previous step
J_temp += J_costtogo[-2, :]

# Save to cost to go matrix
J_costtogo[-1, 0] = np.min(J_temp, axis=0)
l = np.argmin(J_temp, axis=0)

# Save to acc matrix
Input_acc[-1, 0] = acc[l]

# --- Plotting Section ---
# Define arrays for plotting
h_plot = np.zeros(N_h + 1)
v_plot = np.zeros(N_h + 1)
acc_plot = np.zeros(N_h +1)

h_plot[0] = h_init  # Initialize first height
v_plot[0] = v_init  # Initialize first velocity

# Calculate the trajectories based on the optimal policy
for k in range(N_h):
    # Find closest indices in Hd and Vd for height and velocity
    h_plot_index = np.argmin(np.abs(h_plot[k] - Hd))
    v_plot_index = np.argmin(np.abs(v_plot[k] - Vd))

    # Find the corresponding acceleration value from Input_acc
    acc_plot[k] = Input_acc[N_h - k, v_plot_index]

    # Update the next velocity and height using kinematic equations
    v_plot[k + 1] = np.sqrt(v_plot[k]**2 + 2 * acc_plot[k] * (h_max - h_min) / N_h)
    h_plot[k + 1] = h_plot[k] + (h_max - h_min) / N_h

acc_plot[N_h] = np.min(Input_acc[0, :])

# --- Plotting Results ---
# Plot velocity vs height
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(v_plot, h_plot, '--^')
plt.grid(True)
plt.xlabel('Velocity (m/s)')
plt.ylabel('Height (m)')
plt.title('Velocity vs Height')

# Plot acceleration vs height
plt.subplot(2, 1, 2)
plt.plot(acc_plot, h_plot, '^')
plt.grid(True)
plt.xlabel('Acceleration (m/s^2)')
plt.ylabel('Height (m)')
plt.title('Acceleration vs Height')

# Display the plot
plt.tight_layout()
plt.show()
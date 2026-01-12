import numpy as np
import matplotlib.pyplot as plt


# colors = ['tab:cyan', 'tab:purple', 'tab:orange']

data_1 = np.load("results_hallway_v1.npz")['arr_0']
data_2 = np.load("results_hallway_v2.npz")['arr_0']
data_3 = np.load("results_hallway_v3.npz")['arr_0']

# fig, axs = plt.subplots(1, 1, figsize=(7, 7))

# axs.scatter(data_1[:, 1], 1000*data_1[:, 2], color='tab:cyan', label='MPC 1')
# axs.scatter(data_2[:, 1], 1000*data_2[:, 2], color='tab:red', label='MPC 2')
# axs.scatter(data_3[:, 1], 1000*data_3[:, 2], color='tab:orange', label='MPC 3')


fig, ax = plt.subplots(figsize=(7, 6))


ax.errorbar(
    np.mean(data_1[:, 1]), np.mean(1000*data_1[:, 2]),
    xerr=np.std(data_1[:, 1]),
    yerr=np.std(1000*data_1[:, 2]),
    fmt='o',
    capsize=6,
    markersize=8,
    color='tab:cyan',
    label='MPC v1'
)

ax.errorbar(
    np.mean(data_2[:, 1]), np.mean(1000*data_2[:, 2]),
    xerr=np.std(data_2[:, 1]),
    yerr=np.std(1000*data_2[:, 2]),
    fmt='o',
    capsize=6,
    markersize=8,
    color='tab:purple',
    label='MPC v2'
)

ax.errorbar(
    np.mean(data_3[:, 1]), np.mean(1000*data_3[:, 2]),
    xerr=np.std(data_3[:, 1]),
    yerr=np.std(1000*data_3[:, 2]),
    fmt='o',
    capsize=6,
    markersize=8,
    color='tab:orange',
    label='MPC v3'
)



ax.set_title('MPC comparison')
ax.set_xlabel("Average velocity [m/s]")
ax.set_ylabel("RMS Error [mm]")
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

plt.tight_layout()
plt.show()


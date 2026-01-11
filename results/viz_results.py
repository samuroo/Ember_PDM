import numpy as np
import matplotlib.pyplot as plt


data_1 = np.load("results_hallway_v1.npz")['arr_0']
data_2 = np.load("results_hallway_v2.npz")['arr_0']
data_3 = np.load("results_hallway_v3.npz")['arr_0']

fig, axs = plt.subplots(1, 1, figsize=(7, 7))

axs.scatter(data_1[:, 1], 1000*data_1[:, 2], color='tab:cyan', label='MPC 1')
axs.scatter(data_2[:, 1], 1000*data_2[:, 2], color='tab:red', label='MPC 2')
axs.scatter(data_3[:, 1], 1000*data_3[:, 2], color='tab:orange', label='MPC 3')

axs.set_title('Hallway Results')
axs.set_xlabel("Average velocity [m/s]")
axs.set_ylabel("RMS Error [mm]")

axs.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt


data_1 = np.load("results_hallway_v1.npz")['arr_0']
data_2 = np.load("results_hallway_v2.npz")['arr_0']
data_3 = np.load("results_hallway_v3.npz")['arr_0']

fig, axs = plt.subplots(1, 1, figsize=(7, 15))

axs.scatter(data_1[:, 1], data_1[:, 2], color='tab:cyan')
axs.scatter(data_2[:, 1], data_2[:, 2], color='tab:red')
axs.scatter(data_3[:, 1], data_3[:, 2], color='tab:orange')

axs.set_title('Hallway Results')
axs.set_xlabel("Completion velocity [m/s]")
axs.set_ylabel("RMS Error [mm]")

plt.show()
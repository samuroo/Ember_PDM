import numpy as np
import matplotlib.pyplot as plt


labels = ['RRT', 'RRT connect', 'RRT*', 'BIT*']
colors = ['tab:cyan', 'tab:red', 'tab:orange', 'tab:olive']


time = np.array([
    [0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06, 0.05, 0.06, 0.05],     #RRT
    [0.07, 0.07, 0.07, 0.05, 0.07, 0.06, 0.09, 0.05, 0.06, 0.06],     #RRT connect
    [0.13, 0.15, 0.10, 0.13, 0.19, 0.09, 0.11, 0.09, 0.22, 0.12],     #RRT*
    [0.12, 0.14, 0.13, 0.10, 0.12, 0.12, 0.14, 0.08, 0.08, 0.16]      #BIT*
])

length = np.array([
    [10.12, 11.88, 10.24, 9.77, 10.31, 10.42, 10.61, 9.97, 9.62, 10.01],     #RRT
    [10.38, 9.31, 10.25, 9.32, 10.40, 10.31, 11.06, 9.40, 9.61, 10.42],     #RRT connect
    [8.52, 8.93, 8.60, 8.79, 9.57, 8.52, 9.22, 8.27, 9.02, 8.69],     #RRT*
    [8.30, 8.30, 8.36, 8.32, 8.27, 8.41, 8.22, 8.31, 8.24, 9.0]      #BIT*
])

# fig, ax = plt.subplots(figsize=(8, 6))

# for i in range(4):
#     # Scatter points
#     ax.scatter(
#         time[i], length[i],
#         s=70, alpha=0.6,
#         color=colors[i],
#         edgecolor='black',
#         label=labels[i]
#     )

#     # Mean marker
#     ax.scatter(
#         np.mean(time[i]), np.mean(length[i]),
#         s=200, marker='X',
#         color=colors[i], edgecolor='black', zorder=5
#     )

# ax.set_title('Hallway Global Path Planning Comparison', fontsize=14, weight='bold')
# ax.set_xlabel('Planning time [s]')
# ax.set_ylabel('Path length [m]')
# ax.grid(True, linestyle='--', alpha=0.4)
# ax.legend(frameon=True)

# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(7, 6))

for i in range(4):
    ax.errorbar(
        np.mean(time[i]), np.mean(length[i]),
        xerr=np.std(time[i]),
        yerr=np.std(length[i]),
        fmt='o',
        capsize=6,
        markersize=8,
        color=colors[i],
        label=labels[i]
    )

ax.set_title('Mean Planning Performance (±1σ)')
ax.set_xlabel('Planning time [s]')
ax.set_ylabel('Path length [m]')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

plt.tight_layout()
plt.show()

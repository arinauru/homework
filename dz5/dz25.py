import matplotlib.pyplot as plt
import numpy as np

# 1 График

x = [2, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]

plt.figure(figsize=(6, 4))
plt.plot(x, y1, 'r-o', label='line 1')
plt.plot(x, y2, 'g-.o', label='line 1')

plt.legend()
plt.grid(True)
plt.show()

# 2 График

x1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y1 = np.array([0.05, 4.0, 7.0, 6.5, 6.0, 4.5, 3.0, 4.0, 5.5])

x2 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y2 = np.array([9, 7, 4, 3, 2, 3, 4, 7, 9])

x3 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y3 = np.array([-7, -6, -4, -2, 2, -2, -4, -6, -7])

fig = plt.figure(figsize=(8, 6))

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x1, y1, 'b')
ax1.set_xticks(x1)
ax1.set_yticks([2, 4, 6])

ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(x2, y2, 'b')
ax2.set_xticks(x2)
ax2.set_yticks([2, 4, 6, 8])

ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(x3, y3, 'b')
ax3.set_xticks(x3)
ax3.set_yticks([-6, -4, -2, 0, 2])

plt.tight_layout()

plt.show()

# График 3

x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y = np.array([25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25])

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, y, 'b-')

ax.annotate("min", xy=(0, 0), xytext=(0, 5),
            arrowprops=dict(facecolor='black', edgecolor='green', linewidth=4, arrowstyle="->"),
            fontsize=12, color='black', ha="center")

plt.show()

# График 4

data = np.random.randint(0, 11, (8, 8))

fig, ax = plt.subplots()

cax = ax.imshow(data, cmap="viridis", aspect="auto")

cbar = plt.colorbar(cax, ax=ax, shrink=0.5)
cbar.ax.set_anchor((0.5, 0.0))

ax.set_xticks(range(data.shape[1]))
ax.set_yticks(range(data.shape[0]))

ax.invert_yaxis()

plt.show()

# График 5

x = np.linspace(0, 5, 500)
y = np.cos(1 * np.pi * x)

fig, ax = plt.subplots()
ax.plot(x, y, 'r', linewidth=2)
ax.fill_between(x, y, 0, where=(y > 0), color='blue', alpha=0.6)
ax.fill_between(x, y, 0, where=(y < 0), color='blue', alpha=0.6)

ax.set_xlim(0, 5)
ax.set_ylim(-1, 1)

plt.show()

# График 6

x = np.linspace(0, 5, 400)
y = np.cos(1 * np.pi * x)

y[y < -0.5] = np.nan

plt.plot(x, y, linewidth=2)

plt.ylim(-1, 1)
plt.xlim(0, 5)

plt.grid(True)
plt.show()

# График 7

x = np.arange(0, 7, 1)
y = np.arange(0, 7, 1)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].step(x, y, where='pre', color='green', linewidth=2)
axs[0].scatter(x, y, color='green', s=50)

axs[1].step(x, y, where='post', color='green', linewidth=2)
axs[1].scatter(x, y, color='green', s=50)

axs[2].step(x - 0.5, y, where='post', color='green', linewidth=2)
axs[2].scatter(x, y, color='green', s=50)

for ax in axs:
    ax.grid(True)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

plt.show()

# График 8

x = np.linspace(0, 10, 100)

y1 = 5.5 * (1 - ((x - 5) / 5) ** 2)
y2 = 10 * (1 - ((x - 5) / 5) ** 2)
y3 = 5.5 * (1 - ((x - 7) / 5) ** 2) + 4

y1[y1 < 0] = 0
y2[y2 < 0] = 0
y3[y3 < 0] = 0

plt.stackplot(x, y1, y2, y3, labels=['y1', 'y2', 'y3'], colors=['blue', 'orange', 'green'])

plt.ylim(0, 28)
plt.xlim(0, 10)

plt.legend()
plt.show()

# График 9

labels = ['Ford', 'Toyota', 'BMW', 'AUDI', 'Jaguar']
sizes = [25, 15, 55, 19, 36]
colors = ['blue', 'orange', 'green', 'red', 'purple']
explode = [0, 0, 0.1, 0, 0]

plt.pie(sizes, labels=labels, colors=colors, explode=explode, wedgeprops={'edgecolor': 'white'})

plt.show()

# График 10

labels = ['Ford', 'Toyota', 'BMW', 'AUDI', 'Jaguar']
sizes = [25, 15, 55, 19, 36]
colors = ['blue', 'orange', 'green', 'red', 'purple']

plt.pie(sizes, labels=labels, colors=colors, wedgeprops={'edgecolor': 'white', 'width': 0.4})

plt.show()

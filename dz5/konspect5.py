# 1. сценарий
# 2. командная оболочка IPython
# 3. Jupiter
import fig
# 1.
# plt.show() - запускается только 1 раз
# figure

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()

# IPython
# %matplotlib
# import matplotlib.pyplot as plt
# plt.plot()
# pl.draw()

# Jupiter
# %matplotlib inline - в блокнот добавляется статистическая картинка
# %matplotlib notebook - в блокнот добавляются интерактивные графики

fig.savefig('saved_images.png')

print(fig.canvas.get_supported_filetypes())

# Два способа вывода графиков
# - Matlab - подобный стиль
# - в 00 стиле

x = np.linspace(0, 10, 100)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

plt.show()

# fig:Figure, ax:Axes
fig, ax = plt.subplots(2)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

plt.show()

# fig:plt.Figure - контейнер, содержит обьекты (СК, тексты, метки),
# ax:Axes - система координат - прямоугольник, деления, метки

# Цвета линий color
# - 'blue'
# - 'rbgcmyk' -> 'rg'
# - '0.14' - градация серого от 0-1
# - RRGGBB - 'FF00EE'
# - RGB - (1.0, 0.2, 0.3)
# - HTML - 'salmon'

# Стиль линии
# - сплошная '-', 'solid'
# - штриховая '--', 'dashes'
# - штрихпунктирная '-.', 'dashdot'
# - пунктирная ':', 'dotted'

fig = plt.figure()
ax = plt.axes()

ax.plot(x, np.sin(x), color='blue')
ax.plot(x, np.cos(x - 1), linestyle='solid')
ax.plot(x, np.cos(x - 2))
ax.plot(x, np.cos(x - 3))
ax.plot(x, np.cos(x - 4))
ax.plot(x, np.cos(x - 5))
ax.plot(x, np.cos(x - 6), '--k')

plt.show()

fig, ax = plt.subplots(4)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.sin(x))
ax[2].plot(x, np.sin(x))
ax[3].plot(x, np.sin(x))

ax[1].set_xlim(-2, 12)
ax[1].set_ylim(-1.5, 1.5)

ax[2].set_xlim(12, -2)
ax[2].set_ylim(1.5, -1.5)

ax[3].autoscale(tight=True)

plt.subplot(3, 1, 1)
plt.plot(x, np.sin(x))

plt.title('Синус')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.subplot(3, 1, 2)
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')

plt.legend()

plt.subplots_adjust(hspace=0.5)

x = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x), 'o', color='green')

x = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x), '--p', markersize=15,
         linewidth=4, markerfacecolor='white',
         markeredgecolor='gray', markerwidth=2)

rng = np.random.default_rng(0)

colors = rng.random(30)
sizes = 100 * rng.random(30)

plt.scatter(x, np.sin(x), marker='o', c=colors, s=sizes)
plt.colorbar()

# Если точек больше 1000, то plot предпочтительнее из-за производительности

x = np.linspace(0, 10, 50)

dy = 0.4
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.fill_between(x, y - dy, y + dy, color='red', alpha=0.4)

plt.show()


def f(x, y):
    return np.sin(x) ** 5 + np.cos(20 + x * y) * np.cos(x)


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

# plt.contour(X, Y, Z, color='green')
c = plt.contour(X, Y, Z, cmap='RdGy')
plt.clabel(c)

plt.imshow(Z, extent=[0.5, 0.5], cmap='RdGy',
           interpolation='gaussian',
           origin='lower', aspect='equal')
plt.colorbar()

plt.show()

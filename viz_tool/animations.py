import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


frames = 10
data = np.random.rand(frames, 3, 51)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d", )
ax.set_ylim([-1, 1])
ax.set_xlim([-1, 1])
ax.set_zlim([-1, 1])


def animate(i):
    ax.clear()
    ax.scatter(x[i], y[i], z[i])


anim = FuncAnimation(fig, animate, frames=frames, interval=100, repeat=False)
plt.show()

# this doesn't quite do it. 
# - for one i don't save this
# - another issue is the tick marks keep changing so the background is 
#   not fixed
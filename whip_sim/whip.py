import numpy as np

# Import Wrappers
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing

# Import Cosserat Rod Class
from elastica.rod.cosserat_rod import CosseratRod

# Import Boundary Condition Classes
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.external_forces import EndpointForces
from elastica.external_forces import GravityForces, UniformForces, UniformTorques

# Import Timestepping Functions
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate, extend_stepper_interface
import imageio
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
n_elem = 150
density = 1000  # 1000
nu = 0.4  # 0.1
E = 1e8
poisson_ratio = 0.5
# Parameters
start = np.array([0.0, 0.0, 0.0])
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius**2
final_time = 5.0
dl = base_length / n_elem
dt = 0.001 * dl
total_steps = int(final_time / dt)
print("Total steps to take", total_steps)
render_every = 200
DEFAULT_DPI = 100


def render(rod, xlim=(-0.5, 3.5), ylim=(-0.3, 0.3), shape=None):
    if shape is not None:
        figsize = (shape[0] / DEFAULT_DPI, shape[1] / DEFAULT_DPI)
    else:
        figsize = None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(rod.position_collection[2], rod.position_collection[0])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def render2(rod, xlim=(-0.5, 3.5), ylim=(-0.3, 0.3), shape=None):
    if shape is not None:
        figsize = (shape[0] / DEFAULT_DPI, shape[1] / DEFAULT_DPI)
    else:
        figsize = None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(rod.position_collection[2], rod.position_collection[1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


class WhipHandle(FreeRod):
    def __init__(self, center_y=0, amp=0.1, T=1.0):
        FreeRod.__init__(self)
        self.center_y = center_y
        self.amp = amp
        self.T = T

    def constrain_values(self, rod, time):
        y = self.center_y + self.amp * np.sin(2 * np.pi / self.T * time) * np.exp(
            -time / self.T
        )
        rod.position_collection[..., 0] = np.array([y, 0, 0])

    def constrain_rates(self, rod, time):
        vy = (
            self.amp
            * 2
            * np.pi
            / self.T
            * np.cos(2 * np.pi / self.T * time)
            * np.exp(-time / self.T)
        )
        rod.velocity_collection[..., 0] = np.array([vy, 0, 0])


class BeamSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


simulator = BeamSimulator()
# Create rod
my_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    poisson_ratio,
)
simulator.append(my_rod)

# Constrain
simulator.constrain(my_rod).using(
    WhipHandle,
)
# simulator.constrain(my_rod).using(FreeRod)

# Add gravitational forces
gravitational_acc = -9.80665
simulator.add_forcing_to(my_rod).using(
    GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)

simulator.finalize()
timestepper = PositionVerlet()


# integrate(timestepper, simulator, final_time, total_steps)
# imageio.imwrite('output.png', render(my_rod, shape=(128, 128)))
extend_stepper_interface(timestepper, simulator)
time = 0.0
frames = []
for i in tqdm(range(total_steps)):
    time = timestepper.do_step(simulator, time, dt)
    if i % render_every == 0:
        frames.append(render(my_rod, shape=(320, 240)))
frames = np.stack(frames)
imageio.mimwrite("output.gif", frames, fps=10)

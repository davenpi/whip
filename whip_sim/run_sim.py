# import sys
# sys.path.append('/data/vision/billf/scratch/kyi/projects/soft-control/PyElastica')

import os
import argparse
import numpy as np
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import FreeBC
from elastica.external_forces import GravityForces
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import extend_stepper_interface
import imageio
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
import config as cfg

import pdb

GRAVITY_ACC = -9.80665
RENDER_DPI = 100
INIT_START = np.zeros((3,))
INIT_DIR = np.array([0.0, 0.0, 1.0])
INIT_NORMAL = np.array([0.0, 1.0, 0.0])


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=None)
parser.add_argument("--set", nargs="+")
args = parser.parse_args()
config = cfg.get_default()
cfg.set_params(config, args.config_path, args.set)
# pdb.set_trace()
# cfg.freeze(config, save_file=True)


def render(rod, xlim=(-1.5, 1.5), ylim=(-1.0, 1.0), center=(0.0, 0.0), shape=None):
    if shape is not None:
        figsize = (shape[0] / RENDER_DPI, shape[1] / RENDER_DPI)
    else:
        figsize = None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(rod.position_collection[2], rod.position_collection[0])
    ax.plot(
        [center[0], rod.position_collection[2][0]],
        [center[1], rod.position_collection[0][0]],
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.set_aspect(1.0)

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return frame


def get_director(tangent, normal):
    mat = []
    mat.append(normal)
    mat.append(np.cross(tangent, normal))
    mat.append(tangent)
    return np.stack(mat)


class WhipHandlePeriodic(FreeBC):
    def __init__(self, config=config):
        FreeBC.__init__(self)
        self.T = config.handle_period
        self.len_handle = config.len_handle
        self.tilt_angle = config.handle_tilt_angle
        self.swing_angle = config.handle_swing_angle

    def constrain_values(self, rod, time):
        center_theta = self.swing_angle / 2 - self.tilt_angle
        phase = np.arccos(2 * center_theta / self.swing_angle)
        theta = center_theta - self.swing_angle / 2 * np.cos(
            2 * np.pi / self.T * time + phase
        )
        tangent = np.array([np.sin(theta), 0, np.cos(theta)])

        rod.director_collection[..., 0] = get_director(tangent, INIT_NORMAL)
        rod.position_collection[..., 0] = np.array(
            [self.len_handle * np.sin(theta), 0.0, self.len_handle * np.cos(theta)]
        )

    def constrain_rates(self, rod, time):
        center_theta = self.swing_angle / 2 - self.tilt_angle
        phase = np.arccos(2 * center_theta / self.swing_angle)
        theta = center_theta - self.swing_angle / 2 * np.cos(
            2 * np.pi / self.T * time + phase
        )
        omega = (
            self.swing_angle
            * np.pi
            / self.T
            * np.sin(2 * np.pi / self.T * time + phase)
        )

        rod.velocity_collection[..., 0] = (
            omega * self.len_handle * np.array([np.cos(theta), 0, -np.sin(theta)])
        )


class WhipHandlePulse(FreeBC):
    def __init__(self, config=config, N=100):
        FreeBC.__init__(self)

        self.T = config.pulse_period
        self.amp = config.pulse_amp
        self.sigma = config.pulse_sigma
        self.len_handle = config.len_handle
        self.mus = np.array([0.5 * self.T + i * self.T for i in range(N)])

    def constrain_values(self, rod, time):
        # pdb.set_trace()
        theta = (
            self.amp * np.exp(-((time - self.mus) ** 2) / (2 * self.sigma**2)).sum()
        )
        tangent = np.array([np.sin(theta), 0, np.cos(theta)])

        rod.director_collection[..., 0] = get_director(tangent, INIT_NORMAL)
        rod.position_collection[..., 0] = np.array(
            [self.len_handle * np.sin(theta), 0.0, self.len_handle * np.cos(theta)]
        )

    def constrain_rates(self, rod, time):
        theta = (
            self.amp * np.exp(-((time - self.mus) ** 2) / (2 * self.sigma**2)).sum()
        )
        omega = (
            self.amp
            * np.exp(-((time - self.mus) ** 2) / (2 * self.sigma**2))
            * (self.mus - time)
            / self.sigma**2
        ).sum()

        rod.velocity_collection[..., 0] = (
            omega * self.len_handle * np.array([np.cos(theta), 0, -np.sin(theta)])
        )


# class WhipHandleHold(FreeBC):

#     def __init__(self, config=config, )


class BeamSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


simulator = BeamSimulator()


# Create rod
rod_start = (
    np.array(config.center) + np.array(config.init_direction) * config.len_handle
)
whip = CosseratRod.straight_rod(
    config.n_elem,
    rod_start,
    np.array(config.init_direction),
    np.array(config.normal),
    config.base_length,
    config.base_radius,
    config.density,
    config.nu,
    config.E,
    config.poisson_ratio,
)
simulator.append(whip)

if config.handle_motion == "periodic":
    simulator.constrain(whip).using(
        WhipHandlePeriodic,
    )
elif config.handle_motion == "pulse":
    simulator.constrain(whip).using(
        WhipHandlePulse,
    )
else:
    raise ValueError("Invalid handle motion.")

if config.add_gravity:
    simulator.add_forcing_to(whip).using(
        GravityForces, acc_gravity=np.array([GRAVITY_ACC, 0.0, 0.0])
    )

simulator.finalize()

timestepper = PositionVerlet()
extend_stepper_interface(timestepper, simulator)

dl = config.base_length / config.n_elem
dt = config.dt_per_dl * dl
total_steps = int(config.final_time / dt)
render_every = int(1 / (dt * config.render_fps))

time = 0.0
frames = []
for i in tqdm(range(total_steps)):
    time = timestepper.do_step(simulator, time, dt)
    if i % render_every == 0 and config.do_render:
        frames.append(render(whip, shape=(320, 240)))
frames = np.stack(frames)

imageio.mimwrite(
    os.path.join(config.run_dir, "output.gif"), frames, fps=config.render_fps
)

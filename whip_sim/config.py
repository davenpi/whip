import os

import numpy as np
from yacs.config import CfgNode
import pdb

# _PROJ_HOME = ('/data/vision/billf/scratch/kyi/projects/soft-control/PyElastica/kyi')
_PROJ_HOME = "/Users/vidyaraju/Documents/Github/elastica-rl"


def get_default():
    """Get default config."""

    config = CfgNode()

    # Experiment
    config.run_dir = os.path.join(_PROJ_HOME, "outputs/test_run")

    # Whip
    config.n_elem = 100
    config.density = 1000
    config.nu = 0.4
    config.E = 1e7
    config.poisson_ratio = 0.5
    config.base_length = 1.0
    config.base_radius = 0.015
    config.len_handle = 0.2  # length of handle
    config.center = [0.0, 0.0, 0.0]  # rotation center of rod
    config.init_direction = [0.0, 0.0, 1.0]  # init direction of rod
    config.normal = [0.0, 1.0, 0.0]

    # Constraint
    config.handle_motion = "periodic"  # 'periodic' | 'pulse'
    ## Periodic motion
    config.handle_period = 1.0
    config.handle_tilt_angle = np.pi / 12
    config.handle_swing_angle = np.pi

    ## Periodic motion
    config.pulse_amp = 3 * np.pi / 4
    config.pulse_sigma = 0.05
    config.pulse_period = 2.0

    # Environment
    config.add_gravity = True
    config.motion = "translation"  # 'translation' | 'rotation'

    # Simulation
    config.final_time = 5.0
    config.dt_per_dl = 0.005

    # Visualization
    config.do_render = True
    config.render_fps = 30
    config.render_xlim = [-1.5, 1.5]
    config.render_ylim = [-1.0, 1.0]
    config.render_shape = [320, 240]

    return config


def set_params(config, file_path=None, list_opt=None):
    """Set config parameters with config file and options.
    Option list (usually from cammand line) has the highest
    overwrite priority.
    """
    if file_path:
        # if list_opt is None or 'run_dir' not in list_opt[::2]:
        #     raise ValueError('Must specify new run directory.')
        print("- Import config from file {}.".format(file_path))
        config.merge_from_file(file_path)
    if list_opt:
        print("- Overwrite config params {}.".format(str(list_opt[::2])))
        config.merge_from_list(list_opt)
    return config


def freeze(config, save_file=False):
    """Freeze configuration and save to file (optional)."""
    config.freeze()
    if save_file:
        pdb.set_trace()
        if not os.path.isdir(config.run_dir):
            os.makedirs(config.run_dir)
        with open(os.path.join(config.run_dir, "config.yaml"), "w") as fout:
            fout.write(config.dump())

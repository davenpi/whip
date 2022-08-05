from collections import defaultdict
from elastica.wrappers import BaseSystemCollection
from elastica.wrappers import Forcing
from elastica.wrappers import CallBacks
from elastica.wrappers import Constraints
from elastica.boundary_conditions import OneEndFixedBC
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces, UniformForces, EndpointForces
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
import numpy as np
import pickle


class WhipSystem(BaseSystemCollection, Forcing, Constraints, CallBacks):
    """
    Whip simulator.
    """

    pass


whip_simulator = WhipSystem()
# create the timestepper
timestepper = PositionVerlet()
dt = 1e-4
final_time = 1  # time to let the simulation run
total_steps = int(final_time / dt)

# now define my rod. We are treating the whip as a cosserat rod. is this valid?

direction = np.array([0, 0, -1.0])  # tangent direction. why? b/c rod is straight
normal = np.array([0, -1, 0.0], dtype=float)  # normal direction.
rod = CosseratRod.straight_rod(
    n_elements=50,
    start=np.array([0, 0.0, 0]),
    direction=direction,
    normal=normal,
    base_length=1,  # base length (m)
    base_radius=0.1,  # base radius (m)
    density=10,  # density (kg/m^3)
    nu=1e-3,  # energy dissipation of rod. don't know
    youngs_modulus=1e7,  # elastic modulus (Pa)
    shear_modulus=5e-1,  # shear modulus
)
# add the rod to the simulator
whip_simulator.append(rod)

# constrain the rod end
whip_simulator.constrain(rod).using(
    OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# apply forces
start_force = np.array([0, 0.0, 0])
end_force = np.array([0, 0.0001, 0])
whip_simulator.add_forcing_to(rod).using(
    EndpointForces,
    start_force=start_force,
    end_force=end_force,
    ramp_up_time=final_time / 2.0,
)


class WhipCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    # this function is called every time step
    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
        return


# dictionary to store callback data
callback_data = defaultdict(list)

# add callback to system simulator
whip_simulator.collect_diagnostics(rod).using(
    WhipCallBack, step_skip=10, callback_params=callback_data
)

# finalize the simulator
whip_simulator.finalize()

# now let's integrate
integrate(timestepper, whip_simulator, final_time, total_steps)

# save the callback dictionary
f = open("callback.pkl", "wb")
pickle.dump(callback_data, f)
f.close()

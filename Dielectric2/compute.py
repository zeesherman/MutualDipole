# This simple python interface activates the c++ Dielectric2

import math
import numpy as np

# Import the C++ module.
from hoomd.Dielectric2 import _Dielectric2

# Dielectric2 extends a ForceCompute, so we need to bring in the base class
# force and some other parts from hoomd
import hoomd
import hoomd.md
from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.md.force import _force

# The Dielectric2 class.  Computes the forces on each charged dielectric
# particle in an external field with a small, constant gradient.
class Dielectric2(_force):

    # Initialize the Dielectric2 force
    def __init__(self, group, conductivity, field = [0., 0., 0.], gradient = [0., 0., 0.], xi = 0.5, errortol = 1e-3,
		 fileprefix = "", period = 0, dipoleflag = 0):

        hoomd.util.print_status_line();
    
        # initialize base class
        hoomd.md.force._force.__init__(self);

        # Set the cutoff radius for the cell and neighbor lists. This form
        # ensures the calculations are within the specified error tolerance.
        self.rcut = math.sqrt(-math.log(errortol))/xi;      

	# enable the force
        self.enabled = True;

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.error("Sorry, we have not written CPU code for dielectric calculations. \n");
            raise RuntimeError('Error creating Dielectric2');
        else:
            # Create a new neighbor list
            cl_Dielectric2 = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_Dielectric2, "Dielectric2_cl");
            self.neighbor_list = _md.NeighborListGPUBinned(hoomd.context.current.system_definition, self.rcut, 0.4, cl_Dielectric2);
            self.neighbor_list.setEvery(1, True);
            hoomd.context.current.system.addCompute(self.neighbor_list, "Dielectric2_nlist");
            self.neighbor_list.countExclusions();

            # Add the new force to the system
            self.cpp_force = _Dielectric2.Dielectric2(hoomd.context.current.system_definition, group.cpp_group,
                                                      self.neighbor_list, conductivity, field, gradient, xi, errortol, fileprefix,
                                                      period, dipoleflag, hoomd.context.current.system.getCurrentTimeStep());
            hoomd.context.current.system.addCompute(self.cpp_force,self.force_name);

        # Set parameters for the force calculations
        self.cpp_force.SetParams();

        # Compute the forces for the current time step so that they are used for
        # the first update to particle positions.
        self.cpp_force.computeForces(hoomd.context.current.system.getCurrentTimeStep());

    # The integrator calls the update_coeffs function but there are no
    # coefficients to update, so this function does nothing
    def update_coeffs(self):
        pass

    # Update only the external field and gradient.  Faster than the update_parameters function.
    def update_field(self, field, gradient = [0, 0, 0]):
        self.cpp_force.UpdateField(field, gradient);

    # Update simulation parameters.  This is needed if any of the simulation
    # parameters change, including the volume fraction or shape of the simulation box.
    def update_parameters(self, conductivity, field, gradient = [0, 0, 0], fileprefix = "", period = 0, dipoleflag = 0):
        self.cpp_force.UpdateParameters(field, gradient, conductivity, fileprefix, period, dipoleflag,
                                        hoomd.context.current.system.getCurrentTimeStep());
        self.cpp_force.SetParams();




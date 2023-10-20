from hoomd import *
import hoomd
from hoomd import md
from hoomd import MutualDipole

import math
import numpy as np
from datetime import datetime

# Quantities to be specified.
dt = 1e-3  # time step
N = 125000  # number of particles
phi = 0.50  # volume fraction
lambda_p = float("inf") # particle conductivity
error = 1e-3  # desired error tolerance
xi = 0.5  # Ewald splitting parameter
fileprefix = "lambdainf_phi0.50"  # prefix for output files

# Typical nondimensionalization
diameter = 2.  # particle diameter
radius = 1.  # particle radius; sets the length scale to be the particle radius
gamma = 1.  # particle drag coefficient; sets the time scale to be the diffusion time

# Box dimension
L = (4.*math.pi*N/(3.*phi))**(1./3.)

# Create a snapshot of an empty system
context.initialize()
snapshot = data.make_snapshot(N=N, box=data.boxdim(L=L), particle_types=['A'])

# Initialize the system from the snapshot
system = init.read_snapshot(snapshot)

# Parameters for initializing system on a simple cubic lattice
m = int(np.ceil(N**(1./3.)))  # smallest latticle dimension that can hold all of the particles
a = L/m  # distance between lattice sites
lo = -L/2.  # used to center the lattice about the origin

# Initialize a simple cubic array of particles
for p in system.particles:
    (i,j,k) = (p.tag % m, p.tag/m % m, p.tag/m**2  % m)
    p.position = (lo + i*a + a/2, lo + j*a + a/2, lo + k*a + a/2)
    p.type = 'A'
    p.diameter = diameter

# Create neighbor list
nl = md.nlist.cell()

# Function for hard sphere potential
def hs_potential(r, rmin, rmax, dt):
    U = 1./(4.*dt)*(2. - r)**2.
    F = 1./(2.*dt)*(2. - r)
    return(U, F)

# Create an interpolation table for the hard sphere interactions
hs = hoomd.md.pair.table(width=1000, nlist=nl)
hs.pair_coeff.set('A', 'A', func=hs_potential, rmin=0., rmax=2., coeff=dict(dt=dt))

# Establish Brownian dynamics integrator.
all = group.all()
md.integrate.mode_standard(dt=dt)
bd = md.integrate.brownian(group=all, kT=0, seed=datetime.now().microsecond)  # T = 0 turns off Brownian motion
bd.set_gamma('A', gamma=gamma)

# Turn on the field
mutdip = MutualDipole.compute.MutualDipole(group=all, conductivity=[lambda_p]*N, field=[0.0, 0.0, 1.0], xi=xi, errortol=error, fileprefix=fileprefix, period=1, constantdipoleflag=0)

# Run the simulation
run(1)
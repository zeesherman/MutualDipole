#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include <cufft.h>

#ifndef __DIELECTRIC2_CUH__
#define __DIELECTRIC2_CUH__

#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

// Kernel driver for the calculations called by Dielectric2.cc
cudaError_t gpu_ZeroForce(unsigned int Ntotal, // total number of particles
			  Scalar4 *d_force, // pointer to the particle forces 
			  unsigned int block_size); // number of threads per block

cudaError_t gpu_ComputeForce(Scalar4 *d_pos, // particle positions and types
			     int *d_group_membership, // particle membership and index in active group
			     unsigned int Ntotal, // total number of particles
                             unsigned int *d_group_members, // particles in active group
                             unsigned int group_size, // number of particles in active group
                             const BoxDim& box, // simulation box
                             unsigned int block_size, // number of threads per block
			     Scalar4 *d_force, // particle forces 
			     Scalar *d_charge, // particle charges
			     Scalar *d_conductivity, // particle conductivities
			     Scalar3 *d_dipole, // particle dipoles
			     Scalar3 d_extfield, // external field
			     Scalar3 gradient, // external field gradient
			     Scalar3 *d_Eq, // pointer to result of E0 - M_Eq * q
			     Scalar xi, // Ewald splitting parameter
			     Scalar3 eta, // spectral splitting parameter
			     Scalar rc, // real spcae cutoff radius
			     Scalar drtable, // real space table spacing
		     	     int Ntable, // number of entries in the real space tables
			     Scalar2 *d_phiS_table, // potential/dipole real space table
			     Scalar4 *d_ES_table, // field/dipole real space table
			     Scalar2 *d_gradphiq_table, // potential/charge gradient real space table
			     Scalar4 *d_gradphiS_table, // potential/dipole gradient real space table
			     Scalar4 *d_gradES_table, // field/dipole gradient real space table
			     Scalar3 *d_gridk, // wave vector on grid
			     Scalar *d_scale_phiq, // potential/charge scaling on grid
			     Scalar *d_scale_phiS, // potential/dipole scaling on grid
			     Scalar *d_scale_ES, // field/dipole scaling on grid
			     CUFFTCOMPLEX *d_phiq_grid, // potential/charge grid
			     CUFFTCOMPLEX *d_phiS_grid, // potential/dipole grid
			     CUFFTCOMPLEX *d_Eq_gridX, // x component of field/charge grid
			     CUFFTCOMPLEX *d_Eq_gridY, // y component of field/charge grid
			     CUFFTCOMPLEX *d_Eq_gridZ, // z component of field/charge grid
			     CUFFTCOMPLEX *d_ES_gridX, // x component of field/dipole grid
			     CUFFTCOMPLEX *d_ES_gridY, // y component of field/dipole grid
			     CUFFTCOMPLEX *d_ES_gridZ, // z component of field/dipole grid
			     cufftHandle plan, // plan for the FFTs
			     const int Nx, // number of grid nodes in the x dimension
			     const int Ny, // number of grid nodes in the y dimension
			     const int Nz, // number of grid nodes in the z dimension
			     const unsigned int *d_n_neigh, // number of neighbors of each particle
                             const unsigned int *d_nlist, // neighbor list
                             const unsigned int *d_head_list, // used to access entries in the neighbor list
			     int P, // number of grid nodes over which to spread and contract
			     Scalar3 gridh, // grid spacing
			     Scalar errortol, // error tolerance
			     int dipoleflag);  // indicates whether to turn off the mutual dipole functionality or ignore dipoles all together

cudaError_t gpu_ComputeForce_Charge(Scalar4 *d_pos, // particle positions and types
			     int *d_group_membership, // particle membership and index in active group
			     unsigned int Ntotal, // total number of particles
                             unsigned int *d_group_members, // particles in active group
                             unsigned int group_size, // number of particles in active group
                             const BoxDim& box, // simulation box
                             unsigned int block_size, // number of threads per block
			     Scalar4 *d_force, // particle forces 
			     Scalar *d_charge, // particle charges
			     Scalar3 d_extfield, // external field
			     Scalar xi, // Ewald splitting parameter
			     Scalar3 eta, // spectral splitting parameter
			     Scalar rc, // real spcae cutoff radius
			     Scalar drtable, // real space table spacing
		     	     int Ntable, // number of entries in the real space tables
			     Scalar2 *d_gradphiq_table, // potential/charge gradient real space table
			     Scalar *d_scale_phiq, // potential/charge scaling on grid
			     CUFFTCOMPLEX *d_phiq_grid, // potential/charge grid
			     cufftHandle plan, // plan for the FFTs
			     const int Nx, // number of grid nodes in the x dimension
			     const int Ny, // number of grid nodes in the y dimension
			     const int Nz, // number of grid nodes in the z dimension
			     const unsigned int *d_n_neigh, // number of neighbors of each particle
                             const unsigned int *d_nlist, // neighbor list
                             const unsigned int *d_head_list, // used to access entries in the neighbor list
			     int P, // number of grid nodes over which to spread and contract
			     Scalar3 gridh, // grid spacing
			     Scalar errortol); // error tolerance

// Kernel called by PotentialWrapper.cuh
cudaError_t FieldDipoleMultiply(Scalar4 *d_pos, // particle positions and types
			 int *d_group_membership, // particle membership and index in active group
			 unsigned int *d_group_members, // particles in active group
			 unsigned int group_size, // number of particles in active group
			 const BoxDim& box, // simulation box
			 unsigned int block_size, // number of threads to use per block
			 Scalar3 *d_dipole, // particle dipoles
			 Scalar *d_conductivity, // particle conductivities
			 Scalar3 *d_ES, // result of M_ES * S, the dipole contribution to the field
			 Scalar xi, // Ewald splitting parameter
			 Scalar3 eta, // spectral splitting parameter
			 Scalar rc, // real space cutoff radius
			 Scalar drtable, // real space table spacing
			 int Ntable, // number of entries in the real space tables
			 Scalar4 *d_ES_table, // field/dipole real space table
			 Scalar3 *d_gridk, // wave vector on grid
			 Scalar *d_scale_ES, // field/dipole scaling on grid
			 CUFFTCOMPLEX *d_ES_gridX, // x component of field/dipole grid
			 CUFFTCOMPLEX *d_ES_gridY, // y component of field/dipole grid
			 CUFFTCOMPLEX *d_ES_gridZ, // z component of field/dipole grid
			 cufftHandle plan, // plan for the FFTs
			 const int Nx, // number of grid nodes in the x dimension
			 const int Ny, // number of grid nodes in the y dimension
			 const int Nz, // number of grid nodes in the z dimension
			 const unsigned int *d_n_neigh, // number of neighbors of each particle
			 const unsigned int *d_nlist, // neighbor list
			 const unsigned int *d_head_list, // used to access entries in the neighbor list
			 int P, // number of grid nodes over which to spread and contract
			 Scalar3 gridh); // grid spacing

#endif

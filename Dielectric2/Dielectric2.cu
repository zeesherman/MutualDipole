#include "cusp/monitor.h"
#include "cusp/array1d.h"
#include "cusp/krylov/gmres.h"

#include <stdio.h>
#include "Dielectric2.cuh"
#include "PotentialWrapper.cuh"
#include "hoomd/TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define PI 3.1415926535897932f

// Declare textured memory
scalar2_tex_t phiS_table_tex;
scalar4_tex_t ES_table_tex;
scalar2_tex_t gradphiq_table_tex;
scalar4_tex_t gradphiS_table_tex;
scalar4_tex_t gradES_table_tex;
scalar4_tex_t pos_tex;

// Zero particle forces
__global__ void zeroforce(unsigned int Ntotal, // total number of particles
			  Scalar4 *d_force) // pointer to particle forces
{
	// Linear index of current thread
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Set the force to zero
	if (tid < Ntotal) {
		d_force[tid] = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
	}
}

// Zero particle forces (called on the host)
cudaError_t gpu_ZeroForce(unsigned int Ntotal, // total number of particles
			  Scalar4 *d_force, // pointer to particle forces
			  unsigned int block_size) // number of threads per block
{
	// Use one thread per particle
	int Nthreads = block_size;
	int Nblocks = Ntotal/block_size + 1;

	// Call the GPU function to zero forces
	zeroforce<<<Nblocks,Nthreads>>>(Ntotal, d_force);

	return cudaSuccess;
}


// Zero the grid
__global__ void initialize_grid(CUFFTCOMPLEX *grid, // pointer to the grid array
				unsigned int Ngrid) // total number of grid points 
{
	// Linear index of current thread
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Set the grid value to zero
	if (tid < Ngrid) {
		grid[tid] = make_scalar2( 0.0, 0.0 );
	}

}

// Initialize the active group membership/index list
__global__ void initialize_groupmembership( int *d_group_membership, // particle membership and active group index list
				      	    unsigned int Ntotal) // total number of particles
{
	// Global particle index
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	// Flag every particle as not a member of the active group of interest
	if (idx < Ntotal) {
		d_group_membership[idx] = -1;
	}
}

// Determine active group membership and index for all of the particles.  
// A particle with global index i that is not a member of the active group has d_groupmembership[i] = -1.
// A particle with global index i that is a member of the active group has its active group-specific index in d_groupmembership[i].
// That is, d_group_members[d_groupmembership[i]] = i.
__global__ void groupmembership( int *d_group_membership, // particle membership and group index list
				 unsigned int *d_group_members, // group members
				 unsigned int group_size) // number of particles belonging to the group of interest
{
	// Group-specific particle index
	unsigned int group_idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (group_idx < group_size) {
		
		// Global particle index
		unsigned int idx = d_group_members[group_idx];

		// Set the group-specific index at the current particle's global index position in the group membership list
		d_group_membership[idx] = group_idx;
	}

}

// Spread particle charges to a uniform grid.  Use a P-by-P-by-P block per particle with a thread per grid node. 
__global__ void spread_charge( Scalar4 *d_pos, // particle positions
			Scalar *d_charge, // particle dipole moments
			CUFFTCOMPLEX *qgrid, // grid on which to spread the charges
			int group_size, // number of particles belonging to the group of interest
			int Nx, // number of grid nodes in x dimension 
			int Ny, // number of grid nodes in y dimension
			int Nz, // number of grid nodes in z dimension
			unsigned int *d_group_members, // pointer to array of particles belonging to the group
			BoxDim box, // simulation box
			const int P, // number of nodes to spread the particle dipole over
			Scalar3 gridh, // grid spacing
			Scalar3 eta, // Spectral splitting parameter
			Scalar xiterm, // precomputed term 2*xi^2
			Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)
{

	// Setup shared memory for the particle position
	__shared__ float3 shared;
	float3 *pos_shared = &shared;	

	// Get the block index and linear index of the thread within the block
	unsigned int group_idx = blockIdx.x;
	unsigned int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;

	// Global ID of current particle
	unsigned int idx = d_group_members[group_idx];

	// Have the first thread fetch the particle position and store it in shared memory
	if (thread_offset == 0) {

		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Current particle's charge
	Scalar qj = d_charge[group_idx];

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Retrieve position from shared memory
	Scalar3 pos = *pos_shared;

	// Fractional position within box (0 to 1)
	Scalar3 pos_frac = box.makeFraction(pos);

	// Particle position in units of grid spacing
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;

	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).   
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and r^2/eta
	Scalar3 r = pos_grid - pos;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Contribution to the current grid node from the current particle charge
	Scalar charge_inp = prefac*expf( -xiterm*r2eta )*qj;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Add charge contribution to the grid
	atomicAdd( &(qgrid[grid_idx].x), charge_inp);

}

// Spread dipole moments on particles to a uniform grid.  Use a P-by-P-by-P block per particle with a thread per grid node. 
__global__ void spread_dipole( Scalar4 *d_pos, // particle positions
			Scalar3 *d_dipole, // particle dipole moments
			CUFFTCOMPLEX *SgridX, // grid on which to spread the x component of dipoles
			CUFFTCOMPLEX *SgridY, // grid on which to spread the y component of dipoles
			CUFFTCOMPLEX *SgridZ, // grid on which to spread the z component of dipoles
			int group_size, // number of particles belonging to the group of interest
			int Nx, // number of grid nodes in x dimension 
			int Ny, // number of grid nodes in y dimension
			int Nz, // number of grid nodes in z dimension
			unsigned int *d_group_members, // pointer to array of particles belonging to the group
			BoxDim box, // simulation box
			const int P, // number of nodes to spread the particle dipole over
			Scalar3 gridh, // grid spacing
			Scalar3 eta, // Spectral splitting parameter
			Scalar xiterm, // precomputed term 2*xi^2
			Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)
{

	// Setup shared memory for the particle position
	__shared__ float3 shared;
	float3 *pos_shared = &shared;	

	// Get the block index and linear index of the thread within the block
	unsigned int group_idx = blockIdx.x;
	unsigned int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;

	// Global ID of current particle
	unsigned int idx = d_group_members[group_idx];

	// Have the first thread fetch the particle position and store it in shared memory
	if (thread_offset == 0) {

		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Current particle's dipole
	Scalar3 Sj = d_dipole[group_idx];

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Retrieve position from shared memory
	Scalar3 pos = *pos_shared;

	// Fractional position within box (0 to 1)
	Scalar3 pos_frac = box.makeFraction(pos);

	// Particle position in units of grid spacing
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;

	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).   
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and r^2/eta
	Scalar3 r = pos_grid - pos;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Contribution to the current grid node from the current particle dipole
	Scalar3 dipole_inp = prefac*expf( -xiterm*r2eta )*Sj;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Add dipole contribution to the grid
	atomicAdd( &(SgridX[grid_idx].x), dipole_inp.x);
	atomicAdd( &(SgridY[grid_idx].x), dipole_inp.y);
	atomicAdd( &(SgridZ[grid_idx].x), dipole_inp.z);
}

// Spread charge and dipole on particles to a uniform grid.  Use a P-by-P-by-P block per particle with a thread per grid node. 
__global__ void spread( Scalar4 *d_pos, // particle positions
			Scalar *d_charge, // particle charges
			Scalar3 *d_dipole, // particle dipole moments
			CUFFTCOMPLEX *qgrid, // grid on which to spread charges
			CUFFTCOMPLEX *SgridX, // grid on which to spread the x component of dipoles
			CUFFTCOMPLEX *SgridY, // grid on which to spread the y component of dipoles
			CUFFTCOMPLEX *SgridZ, // grid on which to spread the z component of dipoles
			int group_size, // number of particles belonging to the group of interest
			int Nx, // number of grid nodes in x dimension 
			int Ny, // number of grid nodes in y dimension
			int Nz, // number of grid nodes in z dimension
			unsigned int *d_group_members, // pointer to array of particles belonging to the group
			BoxDim box, // simulation box
			const int P, // number of nodes to spread the particle dipole over
			Scalar3 gridh, // grid spacing
			Scalar3 eta, // Spectral splitting parameter
			Scalar xiterm, // precomputed term 2*xi^2
			Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)
{

	// Setup shared memory for the particle position
	__shared__ float3 shared;
	float3 *pos_shared = &shared;	

	// Get the block index and linear index of the thread within the block
	unsigned int group_idx = blockIdx.x;
	unsigned int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;

	// Global ID of current particle
	unsigned int idx = d_group_members[group_idx];

	// Have the first thread fetch the particle position and store it in shared memory
	if (thread_offset == 0) {

		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Current particle's charge and dipole
	Scalar qj = d_charge[group_idx];
	Scalar3 Sj = d_dipole[group_idx];

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Retrieve position from shared memory
	Scalar3 pos = *pos_shared;

	// Fractional position within box (0 to 1)
	Scalar3 pos_frac = box.makeFraction(pos);

	// Particle position in units of grid spacing
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;

	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).   
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and r^2/eta
	Scalar3 r = pos_grid - pos;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Contribution to the current grid node from the current particle charge and dipole
	Scalar charge_inp = prefac*expf( -xiterm*r2eta )*qj;
	Scalar3 dipole_inp = prefac*expf( -xiterm*r2eta )*Sj;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Add dipole contribution to the grid
	atomicAdd( &(qgrid[grid_idx].x), charge_inp);
	atomicAdd( &(SgridX[grid_idx].x), dipole_inp.x);
	atomicAdd( &(SgridY[grid_idx].x), dipole_inp.y);
	atomicAdd( &(SgridZ[grid_idx].x), dipole_inp.z);
}

// Scale the gridded charges to potentials
__global__ void scale_chargepot(CUFFTCOMPLEX *phiq_grid, // potential/charge grid
				Scalar *scale_phiq, // potential/charge scaling on the grid
				unsigned int Ngrid)  // total number of grid nodes
{
  	// Current thread's linear index
  	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	// Ensure thread index corresponds to a node within the grid
  	if ( tid < Ngrid ) {

    	// Read the transformed grid from global memory
	Scalar2 f_phiq_grid = phiq_grid[tid];

    	// Current scaling factor
	Scalar f_phiq = scale_phiq[tid];

    	// Write the scaled grid value to global memory
	phiq_grid[tid] = make_scalar2( f_phiq*f_phiq_grid.x, f_phiq*f_phiq_grid.y );

  	}
}


// Scale the transformed gridded charges to fields
__global__ void scale_chargefield(  	CUFFTCOMPLEX *qgrid, // transform of the gridded charges
					CUFFTCOMPLEX *gridX, // x-component of the scaled grid
					CUFFTCOMPLEX *gridY, // y-component of the scaled grid
					CUFFTCOMPLEX *gridZ, // z-component of the scaled grid
					Scalar3 *gridk,      // wave vector associated with each grid point
					Scalar *scale_phiS, // scaling factor at each grid point
					unsigned int Ngrid)  // total number of grid nodes
{
  	// Current thread's linear index
  	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	// Ensure thread index corresponds to a node within the grid
  	if ( tid < Ngrid ) {

    	// Read the gridded value from global memory
    	Scalar2 fqgrid = qgrid[tid];

    	// Current wave-space vector and scaling
    	Scalar3 k = gridk[tid];
	Scalar f_Eq = -scale_phiS[tid];

    	// Write the scaled grid value to global memory.  The scaling is pure imaginary, so it swaps the .x (real) and .y (imaginary) grid components
	gridX[tid] = make_scalar2( -f_Eq*k.x*fqgrid.y, f_Eq*k.x*fqgrid.x );
	gridY[tid] = make_scalar2( -f_Eq*k.y*fqgrid.y, f_Eq*k.y*fqgrid.x );
	gridZ[tid] = make_scalar2( -f_Eq*k.z*fqgrid.y, f_Eq*k.z*fqgrid.x );

  	}
}

// Scale the transformed gridded dipoles
__global__ void scale_dipole(  CUFFTCOMPLEX *SgridX, // x-component of dipole moments spread onto grid
			CUFFTCOMPLEX *SgridY, // y-component of dipole moments spread onto grid
			CUFFTCOMPLEX *SgridZ, // z-component of dipole moments spread onto grid
			Scalar3 *gridk,      // wave vector associated with each grid node
			Scalar *scale_ES, // field/dipole scaling on the grid
			unsigned int Ngrid)  // total number of grid nodes
{
  	// Current thread's linear index
  	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	// Ensure thread index corresponds to a node within the grid
  	if ( tid < Ngrid ) {

    	// Read the transformed grid from global memory
    	Scalar2 fSgridX = SgridX[tid];
    	Scalar2 fSgridY = SgridY[tid];
    	Scalar2 fSgridZ = SgridZ[tid];

    	// Current wave-space vector and scaling factor
    	Scalar3 k = gridk[tid];
	Scalar f_ES = scale_ES[tid];

    	// Dot product of wave-vector with grid
    	Scalar2 kdotS = (tid==0) ? make_scalar2(0.0,0.0) : make_scalar2( ( k.x*fSgridX.x+k.y*fSgridY.x+k.z*fSgridZ.x ), ( k.x*fSgridX.y+k.y*fSgridY.y+k.z*fSgridZ.y ) );

    	// Write the scaled grid value to global memory
    	SgridX[tid] = make_scalar2( f_ES*k.x*kdotS.x, f_ES*k.x*kdotS.y );
    	SgridY[tid] = make_scalar2( f_ES*k.y*kdotS.x, f_ES*k.y*kdotS.y );
    	SgridZ[tid] = make_scalar2( f_ES*k.z*kdotS.x, f_ES*k.z*kdotS.y );

  	}
}

// Scale the gridded charges and dipoles
__global__ void scale(  CUFFTCOMPLEX *phiq_grid, // potential/charge grid
			CUFFTCOMPLEX *phiS_grid, // potential/dipole grid
			CUFFTCOMPLEX *Eq_gridX, // x-component of field/charge grid
			CUFFTCOMPLEX *Eq_gridY, // y-component of field/charge grid
			CUFFTCOMPLEX *Eq_gridZ, // z-component of field/charge grid
			CUFFTCOMPLEX *ES_gridX, // x-component of field/dipole grid
			CUFFTCOMPLEX *ES_gridY, // y-component of field/dipole grid
			CUFFTCOMPLEX *ES_gridZ, // z-component of field/dipole grid
			Scalar3 *gridk,      // wave vector associated with each grid node
			Scalar *scale_phiq, // potential/charge scaling on the grid
			Scalar *scale_phiS, // potential/dipole scaling on the grid
			Scalar *scale_ES, // field/dipole scaling on the grid
			unsigned int Ngrid)  // total number of grid nodes
{
  	// Current thread's linear index
  	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	// Ensure thread index corresponds to a node within the grid
  	if ( tid < Ngrid ) {

    	// Read the transformed grid from global memory
	Scalar2 f_phiq_grid = phiq_grid[tid];
    	Scalar2 f_ES_gridX = ES_gridX[tid];
    	Scalar2 f_ES_gridY = ES_gridY[tid];
    	Scalar2 f_ES_gridZ = ES_gridZ[tid];

    	// Current wave-space vector and scaling factors
    	Scalar3 k = gridk[tid];
	Scalar f_phiq = scale_phiq[tid];
	Scalar f_phiS = scale_phiS[tid];
	Scalar f_ES = scale_ES[tid];

    	// Dot product of wave-vector with grid
    	Scalar2 kdotS = (tid==0) ? make_scalar2(0.0,0.0) : make_scalar2( ( k.x*f_ES_gridX.x+k.y*f_ES_gridY.x+k.z*f_ES_gridZ.x ), ( k.x*f_ES_gridX.y+k.y*f_ES_gridY.y+k.z*f_ES_gridZ.y ) );

    	// Write the scaled grid value to global memory
	phiq_grid[tid] = make_scalar2( f_phiq*f_phiq_grid.x, f_phiq*f_phiq_grid.y );
	phiS_grid[tid] = make_scalar2( -f_phiS*kdotS.y, f_phiS*kdotS.x );
	Eq_gridX[tid] = make_scalar2( f_phiS*k.x*f_phiq_grid.y, -f_phiS*k.x*f_phiq_grid.x );
	Eq_gridY[tid] = make_scalar2( f_phiS*k.y*f_phiq_grid.y, -f_phiS*k.y*f_phiq_grid.x );
	Eq_gridZ[tid] = make_scalar2( f_phiS*k.z*f_phiq_grid.y, -f_phiS*k.z*f_phiq_grid.x );
    	ES_gridX[tid] = make_scalar2( f_ES*k.x*kdotS.x, f_ES*k.x*kdotS.y );
    	ES_gridY[tid] = make_scalar2( f_ES*k.y*kdotS.x, f_ES*k.y*kdotS.y );
    	ES_gridZ[tid] = make_scalar2( f_ES*k.z*kdotS.x, f_ES*k.z*kdotS.y );

  	}
}

// Contract the grid to the particle centers
__global__ void contract(	Scalar4 *d_pos,  // particle positions
				Scalar3 *d_output, // pointer to the output array
				CUFFTCOMPLEX *gridX, // x component of grid
				CUFFTCOMPLEX *gridY, // y component of grid
				CUFFTCOMPLEX *gridZ, // z component of grid 
				int group_size, // number of particles belonging to the group of interest
				int Nx, // number of grid nodes in x dimension
				int Ny, // number of grid nodes in y dimension
				int Nz, // number of grid nodes in z dimension
				unsigned int *d_group_members, // pointer to array of particles belonging to the group
				BoxDim box, // simulation box
				const int P, // number of nodes to spread the particle dipole over
				Scalar3 gridh, // grid spacing
				Scalar xi, //  Ewald splitting parameter
				Scalar3 eta, // Spectral splitting parameter
				Scalar xiterm, // precomputed term 2*xi^2
				Scalar prefac) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle output and particle positions
	extern __shared__ float3 shared[];  // 16 kb maximum
	float3 *output = shared;
	float3 *pos_shared = &shared[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int blocksize = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    	unsigned int idx = d_group_members[group_idx];

	// Initialize the shared memory and have the first thread fetch the particle position and store it in shared memory
	output[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if (thread_offset == 0){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = pos_shared[0];

    	// Express the particle position in units of grid spacing.
    	Scalar3 pos_frac = box.makeFraction(pos);
    	pos_frac.x *= (Scalar)Nx;
    	pos_frac.y *= (Scalar)Ny;
    	pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and "r^2/eta"
	Scalar3 r = pos - pos_grid;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Contribution to the current particle from the current grid node
	Scalar3 gridXYZ = make_scalar3(gridX[grid_idx].x, gridY[grid_idx].x, gridZ[grid_idx].x);
	output[thread_offset] = prefac*expf( -xiterm*r2eta )*gridXYZ;	

	// Reduction to add all of the P^3 values
	int offs = blocksize;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			output[thread_offset] += output[thread_offset + offs];
		}
	}

	if (thread_offset == 0){
		// Store the current particle's output
		d_output[group_idx] = output[0];
	}
}

// Contract the charge-only grid to forces on particles
__global__ void contract_force_charge(	Scalar4 *d_pos,  // particle positions
					Scalar *d_charge, // particle charges
					Scalar4 *d_force, // pointer to particle forces
					CUFFTCOMPLEX *phiq_grid, // potential/charge grid
					int group_size, // number of particles belonging to the group of interest (group = all is all particles)
					int Nx, // number of grid nodes in x dimension
					int Ny, // number of grid nodes in y dimension
					int Nz, // number of grid nodes in z dimension
					unsigned int *d_group_members, // pointer to array of particles belonging to the group
					BoxDim box, // simulation box
					const int P, // number of nodes to spread the particle dipole over
					Scalar3 gridh, // grid spacing
					Scalar3 eta, // Spectral splitting parameter
					Scalar xiterm, // precomputed term 2*xi^2
					Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle forces and particle positions
	extern __shared__ float3 shared[];  // 16 kb maximum
	float3 *force = shared;
	float3 *pos_shared = &shared[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int blocksize = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    	unsigned int idx = d_group_members[group_idx];

	// Initialize the shared memory and have the first thread fetch the particle position and store it in shared memory
	force[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if (thread_offset == 0){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Current particle's charge
	Scalar qi = d_charge[group_idx];

	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = pos_shared[0];

    	// Express the particle position in units of grid spacing.
    	Scalar3 pos_frac = box.makeFraction(pos);
    	pos_frac.x *= (Scalar)Nx;
    	pos_frac.y *= (Scalar)Ny;
    	pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and combinations with eta
	Scalar3 r = pos - pos_grid;
	Scalar3 reta = r/eta;
	Scalar r2eta = r.x*reta.x + r.y*reta.y + r.z*reta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Product of the current charge with the current grid value
	Scalar qi_dot_phiq_grid = qi*phiq_grid[grid_idx].x;

	// Contribution to the current particle from the current grid node
	force[thread_offset] = 2.0*prefac*xiterm*qi_dot_phiq_grid*expf( -xiterm*r2eta )*reta;

	// Reduction to add all of the P^3 values
	int offs = blocksize;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			force[thread_offset] += force[thread_offset + offs];
		}
	}

	if (thread_offset == 0){
		d_force[idx] = make_scalar4(force[0].x, force[0].y, force[0].z, 0.0);
	}
}


// Contract the grid to forces on particles
__global__ void contract_force(	Scalar4 *d_pos,  // particle positions
				Scalar *d_charge, // particle charges
				Scalar3 *d_dipole, // particle dipole moments
				Scalar4 *d_force, // pointer to particle forces
				CUFFTCOMPLEX *phiq_grid, // potential/charge grid
				CUFFTCOMPLEX *phiS_grid, // potential/dipole grid
				CUFFTCOMPLEX *Eq_gridX, // x component of field/charge grid
				CUFFTCOMPLEX *Eq_gridY, // y component of field/charge grid
				CUFFTCOMPLEX *Eq_gridZ, // z component of field/charge grid 
				CUFFTCOMPLEX *ES_gridX, // x component of field/dipole grid
				CUFFTCOMPLEX *ES_gridY, // y component of field/dipole grid
				CUFFTCOMPLEX *ES_gridZ, // z component of field/dipole grid 
				int group_size, // number of particles belonging to the group of interest (group = all is all particles)
				int Nx, // number of grid nodes in x dimension
				int Ny, // number of grid nodes in y dimension
				int Nz, // number of grid nodes in z dimension
				unsigned int *d_group_members, // pointer to array of particles belonging to the group
				BoxDim box, // simulation box
				const int P, // number of nodes to spread the particle dipole over
				Scalar3 gridh, // grid spacing
				Scalar3 eta, // Spectral splitting parameter
				Scalar xiterm, // precomputed term 2*xi^2
				Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle forces and particle positions
	extern __shared__ float3 shared[];  // 16 kb maximum
	float3 *force = shared;
	float3 *pos_shared = &shared[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int blocksize = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    	unsigned int idx = d_group_members[group_idx];

	// Initialize the shared memory and have the first thread fetch the particle position and store it in shared memory
	force[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if (thread_offset == 0){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Current particle's charge and dipole
	Scalar qi = d_charge[group_idx];
	Scalar3 Si = d_dipole[group_idx];

	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = pos_shared[0];

    	// Express the particle position in units of grid spacing.
    	Scalar3 pos_frac = box.makeFraction(pos);
    	pos_frac.x *= (Scalar)Nx;
    	pos_frac.y *= (Scalar)Ny;
    	pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and combinations with eta
	Scalar3 r = pos - pos_grid;
	Scalar3 reta = r/eta;
	Scalar r2eta = r.x*reta.x + r.y*reta.y + r.z*reta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Product of the current charge and dipole with the current grid values
	Scalar qi_dot_phiq_grid = qi*phiq_grid[grid_idx].x;
	Scalar qi_dot_phiS_grid = qi*phiS_grid[grid_idx].x;
	Scalar Si_dot_Eq_grid = Si.x*Eq_gridX[grid_idx].x + Si.y*Eq_gridY[grid_idx].x + Si.z*Eq_gridZ[grid_idx].x;
	Scalar Si_dot_ES_grid = Si.x*ES_gridX[grid_idx].x + Si.y*ES_gridY[grid_idx].x + Si.z*ES_gridZ[grid_idx].x;

	// Contribution to the current particle from the current grid node
	force[thread_offset] = 2.0*prefac*xiterm*(qi_dot_phiq_grid + qi_dot_phiS_grid + Si_dot_Eq_grid + Si_dot_ES_grid)*expf( -xiterm*r2eta )*reta;

	// Reduction to add all of the P^3 values
	int offs = blocksize;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			force[thread_offset] += force[thread_offset + offs];
		}
	}

	if (thread_offset == 0){
		d_force[idx] = make_scalar4(force[0].x, force[0].y, force[0].z, 0.0);
	}
}

// Add real space contribution to particle field
__global__ void real_space_field_charge( 	Scalar4 *d_pos, // particle positions and types
					Scalar *d_charge, // particle charges
					Scalar3 E0, // external electric field
					Scalar3 *d_Eq, // pointer to result of E0 - M_Eq * q
					int group_size, // number of particles in the group of which the field is being calculated
					Scalar2 *d_phiS_table, // real space potential/dipole table
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					int *d_group_membership, // particle membership and index in group
					unsigned int *d_group_members, // pointer to array of particles belonging to the group
					BoxDim box, // simulation box
					const unsigned int *d_n_neigh, // pointer to the number of neighbors of each particle 
					const unsigned int *d_nlist, // pointer to the neighbor list 
					const unsigned int *d_head_list) // index used to access elements of the neighbor list
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];

		// Get the wave space contribution to E0 - M_Eq * q
  		Scalar3 Eq = d_Eq[group_idx];

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = texFetchScalar4(d_pos, pos_tex, idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {
			
			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			int neigh_group_idx = d_group_membership[neigh_idx];

			// Check if neighbor is a member of the group of interest
			if ( neigh_group_idx != -1 ) {

				// Position and type of neighbor particle
				Scalar4 postypej = texFetchScalar4(d_pos, pos_tex, neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Charge of neighbor particle
					Scalar qj = d_charge[neigh_group_idx];
					
					// Read the table values closest to the current distance
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );	
					Scalar2 entry = texFetchScalar2(d_phiS_table, phiS_table_tex, tableind);

					// Linearly interpolate between the table values
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar A = entry.x + ( entry.y - entry.x )*lininterp; 

					// Real-space contributions to M_Eq * q. Because the table values are for the potential/dipole coupling, we negate it to get the field/charge coupling.
					Eq += -A*qj*r;
	
      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Subtract the result from the external field and write to the current particle's position in the output array
		d_Eq[group_idx] = E0 - Eq;

	}
}

// Add real space contribution to particle field
__global__ void real_space_field_dipole( 	Scalar4 *d_pos, // particle positions and types
					Scalar3 *d_dipole, // particle dipoles
					Scalar *d_conductivity, // particle conductivities
					Scalar3 *d_ES, // pointer to dipole contribution to the field
					int group_size, // number of particles in the group of which the field is being calculated
					Scalar4 *d_ES_table, // field/dipole real space table
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					int *d_group_membership, // particle membership and index in group
					unsigned int *d_group_members, // pointer to array of particles belonging to the group
					BoxDim box, // simulation box
					const unsigned int *d_n_neigh, // pointer to the number of neighbors of each particle 
					const unsigned int *d_nlist, // pointer to the neighbor list 
					const unsigned int *d_head_list, // index used to access elements of the neighbor list
					Scalar selfcoeff) // coefficient of the self term
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];

		// Get the wave space contribution to M_ES * S
  		Scalar3 ES = d_ES[group_idx];

		// Dipole moment and conductivity of current particle
		Scalar3 Si = d_dipole[group_idx];
		Scalar lambda_p = d_conductivity[group_idx];
		
		// Add real space self term
		ES += selfcoeff*Si;

		// If the particle conductivity is finite, add an additional self term
		if ( isfinite(lambda_p) ) {
			ES += 3.0/(4.0*PI*(lambda_p - 1.0))*Si;
		}

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = texFetchScalar4(d_pos, pos_tex, idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {
			
			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			int neigh_group_idx = d_group_membership[neigh_idx];

			// Check if neighbor is a member of the group of interest
			if ( neigh_group_idx != -1 ) {

				// Position and type of neighbor particle
				Scalar4 postypej = texFetchScalar4(d_pos, pos_tex, neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Dipole of neighbor particle
					Scalar3 Sj = d_dipole[neigh_group_idx];

					// Dot product of neighbor dipole and r
					Scalar Sjdotr = Sj.x*r.x + Sj.y*r.y + Sj.z*r.z;

					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );	
					Scalar4 entry = texFetchScalar4(d_ES_table, ES_table_tex, tableind);

					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar C1 = entry.x + ( entry.z - entry.x )*lininterp;
					Scalar C2 = entry.y + ( entry.w - entry.y )*lininterp;  

					// Real-space contributions to M_ES * S
					ES += C1*(Sj - Sjdotr*r) + C2*Sjdotr*r;
	
      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Write the result to the current particle's output
		d_ES[group_idx] = ES;

	}
}

// Add real space contribution to charge-only particle forces
__global__ void real_space_force_charge(Scalar4 *d_pos, // particle positions and types
					Scalar *d_charge, // particle charges
					Scalar4 *d_force, // pointer to particle forces
					Scalar3 E0, // external field
					int group_size, // number of particles in the group of which the force is being calculated
					Scalar2 *d_gradphiq_table, // potential/charge gradient real space table
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					int *d_group_membership, // particle membership and index in group
					unsigned int *d_group_members, // pointer to array of particles belonging to the group
					BoxDim box, // simulation box
					const unsigned int *d_n_neigh, // pointer to the number of neighbors of each particle 
					const unsigned int *d_nlist, // pointer to the neighbor list 
					const unsigned int *d_head_list) // index used to access elements of the neighbor list
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];

		// Get the reciprocal contribution to the force
  		Scalar4 F4 = d_force[idx];
		Scalar3 F = make_scalar3(F4.x, F4.y, F4.z);

		// Charge and dipole of current particle
		Scalar qi = d_charge[group_idx];

		// Add the phoretic forces
		F += qi*E0; // electrophoretic force

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = texFetchScalar4(d_pos, pos_tex, idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {

			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			int neigh_group_idx = d_group_membership[neigh_idx];

			// Check if neighbor is a member of the group of interest
			if ( neigh_group_idx != -1 ) {

				// Position and type of neighbor particle
				Scalar4 postypej = texFetchScalar4(d_pos, pos_tex, neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Charge of neighbor particle
					Scalar qj = d_charge[neigh_group_idx];
	
					// Find the entries in the real space tables between which to interpolate
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );
					Scalar2 gradphiq_entry = texFetchScalar2(d_gradphiq_table, gradphiq_table_tex, tableind);	

					// Interpolate between the values in the tables
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar A = gradphiq_entry.x + ( gradphiq_entry.y - gradphiq_entry.x )*lininterp;

					// Real-space contributions to the force
					F += -A*qi*qj*r; // charge/charge

      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Write the result to the current particle's force
		F4 = make_scalar4(F.x, F.y, F.z, 0.0);
		d_force[idx] = F4;
	}
}


// Add real space contribution to particle forces
__global__ void real_space_force( 	Scalar4 *d_pos, // particle positions and types
					Scalar *d_charge, // particle charges
					Scalar3 *d_dipole, // particle dipoles
					Scalar4 *d_force, // pointer to particle forces
					Scalar3 E0, // external field
					Scalar3 gradient, // external field gradient
					int group_size, // number of particles in the group of which the force is being calculated
					Scalar2 *d_gradphiq_table, // potential/charge gradient real space table
					Scalar4 *d_gradphiS_table, // potential/dipole gradient real space tbale
					Scalar4 *d_gradES_table, // field/dipole gradient real space table
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					int *d_group_membership, // particle membership and index in group
					unsigned int *d_group_members, // pointer to array of particles belonging to the group
					BoxDim box, // simulation box
					const unsigned int *d_n_neigh, // pointer to the number of neighbors of each particle 
					const unsigned int *d_nlist, // pointer to the neighbor list 
					const unsigned int *d_head_list) // index used to access elements of the neighbor list
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];

		// Get the reciprocal contribution to the force
  		Scalar4 F4 = d_force[idx];
		Scalar3 F = make_scalar3(F4.x, F4.y, F4.z);

		// Charge and dipole of current particle
		Scalar qi = d_charge[group_idx];
		Scalar3 Si = d_dipole[group_idx];

		// Add the phoretic forces
		Scalar E0_mag = sqrtf(E0.x*E0.x + E0.y*E0.y + E0.z*E0.z); // field magnitude
		F += qi*E0; // electrophoretic force
		if (E0_mag >= 1e-6){
			F += gradient*(Si.x*E0.x + Si.y*E0.y + Si.z*E0.z)/E0_mag; // dielectrophoretic force
		}

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = texFetchScalar4(d_pos, pos_tex, idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {

			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			int neigh_group_idx = d_group_membership[neigh_idx];

			// Check if neighbor is a member of the group of interest
			if ( neigh_group_idx != -1 ) {

				// Position and type of neighbor particle
				Scalar4 postypej = texFetchScalar4(d_pos, pos_tex, neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Charge and dipole of neighbor particle
					Scalar qj = d_charge[neigh_group_idx];
					Scalar3 Sj = d_dipole[neigh_group_idx];

					// Dot products of the two dipoles
					Scalar SidotSj = Si.x*Sj.x + Si.y*Sj.y + Si.z*Sj.z;
					Scalar Sidotr = Si.x*r.x + Si.y*r.y + Si.z*r.z;
					Scalar Sjdotr = Sj.x*r.x + Sj.y*r.y + Sj.z*r.z;
	
					// Find the entries in the real space tables between which to interpolate
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );
					Scalar2 gradphiq_entry = texFetchScalar2(d_gradphiq_table, gradphiq_table_tex, tableind);	
					Scalar4 gradphiS_entry = texFetchScalar4(d_gradphiS_table, gradphiS_table_tex, tableind);
					Scalar4 gradES_entry = texFetchScalar4(d_gradES_table, gradES_table_tex, tableind);

					// Interpolate between the values in the tables
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar A = gradphiq_entry.x + ( gradphiq_entry.y - gradphiq_entry.x )*lininterp;
					Scalar B = gradphiS_entry.x + ( gradphiS_entry.z - gradphiS_entry.x )*lininterp;
					Scalar C = gradphiS_entry.y + ( gradphiS_entry.w - gradphiS_entry.y )*lininterp;
					Scalar D = gradES_entry.x + ( gradES_entry.z - gradES_entry.x )*lininterp;
					Scalar E = gradES_entry.y + ( gradES_entry.w - gradES_entry.y )*lininterp;  

					// Real-space contributions to the force
					F += -A*qi*qj*r; // charge/charge
					F += -qi*(B*(Sj - Sjdotr*r) + C*Sjdotr*r); // charge/dipole
					F += qj*(B*(Si - Sidotr*r) + C*Sidotr*r); // dipole/charge 
					F += -D*(SidotSj*r + Sjdotr*Si + Sidotr*Sj) + (2.0*D-E)*Sidotr*Sjdotr*r; // dipole/dipole

      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Write the result to the current particle's force
		F4 = make_scalar4(F.x, F.y, F.z, 0.0);
		d_force[idx] = F4;
	}
}

cudaError_t FieldChargeMultiply(       Scalar4 *d_pos, // particle posisitons
				int *d_group_membership, // particle membership and index in active group
				unsigned int *d_group_members, // particles in active group
				unsigned int group_size, // number of particles in active group
				const BoxDim& box, // simulation box
				unsigned int block_size, // number of threads to use per block
				Scalar *d_charge, // particle charge
				Scalar3 *d_Eq, // pointer to result of E0 - M_Eq * q
				Scalar3 extfield, // external field
				Scalar xi, // Ewald splitting parameter
				Scalar3 eta, // spectral splitting parameter
				Scalar rc, // real space cutoff radius
				Scalar drtable, // real space coefficient table spacing
				int Ntable, // number of entries in the real space coefficient table
				Scalar2 *d_phiS_table, // pointer to field/charge real space table
				Scalar3 *d_gridk, // wave vector on grid
				Scalar *d_scale_phiS, // potential/dipole scaling on grid
				CUFFTCOMPLEX *d_qgrid, // wave space charge grid
				CUFFTCOMPLEX *d_SgridX, // x component of dipole grid
				CUFFTCOMPLEX *d_SgridY, // y component of dipole grid
				CUFFTCOMPLEX *d_SgridZ, // z component of dipole grid
				cufftHandle plan, // plan for the FFTs
				const int Nx, // number of grid nodes in the x dimension
				const int Ny, // number of grid nodes in the y dimension
				const int Nz, // number of grid nodes in the z dimension
				const unsigned int *d_n_neigh, // number of neighbors of each particle
				const unsigned int *d_nlist, // neighbor list
				const unsigned int *d_head_list, // used to access entries in the neighbor list
				int P, // number of grid nodes over which to spread and contract
				Scalar3 gridh) // grid spacing
{
	// total number of grid nodes
 	unsigned int Ngrid = Nx*Ny*Nz;

	// for initialization and scaling, use one thread per grid node
    	int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    	int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block per particle.
	dim3 Nblocks2(group_size, 1, 1); // grid is a 1-D array of N blocks, where N is number of particles
	dim3 Nthreads2(P, P, P); // block is 3-D array of P^3 threads

    	// for the real space calculation, use one thread per particle
    	dim3 Nblocks3( (group_size/block_size) + 1, 1, 1);
    	dim3 Nthreads3(block_size, 1, 1);

	// Factors needed for the kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xi2 = xi*xi;
	Scalar xi3 = xi2*xi;
	Scalar xiterm = 2.0*xi2;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials

    	// Reset the grid values to zero
	initialize_grid<<<Nblocks1, Nthreads1>>>(d_qgrid,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_SgridX,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_SgridY,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_SgridZ,Ngrid);

	// Spread charges from the particles to the grid
	spread_charge<<<Nblocks2, Nthreads2>>>(d_pos, d_charge, d_qgrid, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, eta, xiterm, prefac);

	//  Compute the Fourier transform of the gridded data
    	cufftExecC2C(plan, d_qgrid, d_qgrid, CUFFT_FORWARD);

	// Scale the grid values
    	scale_chargefield<<<Nblocks1, Nthreads1>>>(d_qgrid, d_SgridX, d_SgridY, d_SgridZ, d_gridk, d_scale_phiS, Ngrid);

	// Inverse Fourier transform the gridded data
    	cufftExecC2C(plan, d_SgridX, d_SgridX, CUFFT_INVERSE);
	cufftExecC2C(plan, d_SgridY, d_SgridY, CUFFT_INVERSE);
	cufftExecC2C(plan, d_SgridZ, d_SgridZ, CUFFT_INVERSE);

	// Contract the gridded values to the particles to get the wave space contribution to the field
	contract<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_pos, d_Eq, d_SgridX, d_SgridY, d_SgridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, xi, eta, xiterm, quadW*prefac);   

	gpuErrchk(cudaPeekAtLastError());

	// Compute the real space contribution to the field
    	real_space_field_charge<<<Nblocks3, Nthreads3>>>(d_pos, d_charge, extfield, d_Eq, group_size, d_phiS_table, rc, Ntable, drtable, d_group_membership, d_group_members, box, d_n_neigh, d_nlist, d_head_list); 

    	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

cudaError_t FieldDipoleMultiply(       Scalar4 *d_pos, // particle posisitons
				int *d_group_membership, // particle membership and index in active group
				unsigned int *d_group_members, // particles in active group
				unsigned int group_size, // number of particles in active group
				const BoxDim& box, // simulation box
				unsigned int block_size, // number of threads to use per block
				Scalar3 *d_dipole, // particle dipoles
				Scalar *d_conductivity, // particle conductivity
				Scalar3 *d_ES, // pointer to dipole contribution to the field
				Scalar xi, // Ewald splitting parameter
				Scalar3 eta, // spectral splitting parameter
				Scalar rc, // real space cutoff radius
				Scalar drtable, // real space coefficient table spacing
				int Ntable, // number of entries in the real space coefficient table
				Scalar4 *d_ES_table, // pointer to field/dipole real space table
				Scalar3 *d_gridk, // wave vector on grid
				Scalar *d_scale_ES, // field/dipole scaling on grid
				CUFFTCOMPLEX *d_SgridX, // x component of wave space grid
				CUFFTCOMPLEX *d_SgridY, // y component of wave space grid
				CUFFTCOMPLEX *d_SgridZ, // z component of wave space grid
				cufftHandle plan, // plan for the FFTs
				const int Nx, // number of grid nodes in the x dimension
				const int Ny, // number of grid nodes in the y dimension
				const int Nz, // number of grid nodes in the z dimension
				const unsigned int *d_n_neigh, // number of neighbors of each particle
				const unsigned int *d_nlist, // neighbor list
				const unsigned int *d_head_list, // used to access entries in the neighbor list
				int P, // number of grid nodes over which to spread and contract
				Scalar3 gridh) // grid spacing
{
	// total number of grid nodes
 	unsigned int Ngrid = Nx*Ny*Nz;

	// for initialization and scaling, use one thread per grid node
    	int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    	int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block per particle.
	dim3 Nblocks2(group_size, 1, 1); // grid is a 1-D array of N blocks, where N is number of particles
	dim3 Nthreads2(P, P, P); // block is 3-D array of P^3 threads

    	// for the real space calculation, use one thread per particle
    	dim3 Nblocks3( (group_size/block_size) + 1, 1, 1);
    	dim3 Nthreads3(block_size, 1, 1);

	// Factors needed for the kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xi2 = xi*xi;
	Scalar xi3 = xi2*xi;
	Scalar xiterm = 2.0*xi2;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials
	Scalar selfterm = (-1.0+6.0*xi2)/(16.0*PI*sqrt(PI)*xi3) + (1.0 - 2.0*xi2)*exp(-4.0*xi2)/(16.0*PI*sqrt(PI)*xi3) + erfc(2.0*xi)/(4.0*PI); // self term in the potential matrix

    	// Reset the grid values to zero
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_SgridX,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_SgridY,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_SgridZ,Ngrid);

	// Spread dipoles from the particles to the grid
	spread_dipole<<<Nblocks2, Nthreads2>>>(d_pos, d_dipole, d_SgridX, d_SgridY, d_SgridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, eta, xiterm, prefac);

	//  Compute the Fourier transform of the gridded data
    	cufftExecC2C(plan, d_SgridX, d_SgridX, CUFFT_FORWARD);
    	cufftExecC2C(plan, d_SgridY, d_SgridY, CUFFT_FORWARD);
    	cufftExecC2C(plan, d_SgridZ, d_SgridZ, CUFFT_FORWARD);

	// Scale the grid values
    	scale_dipole<<<Nblocks1, Nthreads1>>>(d_SgridX, d_SgridY, d_SgridZ, d_gridk, d_scale_ES, Ngrid);

	// Inverse Fourier transform the gridded data
    	cufftExecC2C(plan, d_SgridX, d_SgridX, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_SgridY, d_SgridY, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_SgridZ, d_SgridZ, CUFFT_INVERSE);

	// Contract the gridded values to the particles to get the wave space contribution to M_ES * S
	contract<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_pos, d_ES, d_SgridX, d_SgridY, d_SgridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, xi, eta, xiterm, quadW*prefac);   

	gpuErrchk(cudaPeekAtLastError());

	// Compute the real space contribution to M_ES * S
    	real_space_field_dipole<<<Nblocks3, Nthreads3>>>(d_pos, d_dipole, d_conductivity, d_ES, group_size, d_ES_table, rc, Ntable, drtable, d_group_membership, d_group_members, box, d_n_neigh, d_nlist, d_head_list, selfterm); 

    	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

// Compute the particle dipoles iteratively using GMRES
cudaError_t ComputeDipole(	Scalar4 *d_pos, // particle posisitons
				int *d_group_membership, // particle membership and index in active group
				unsigned int *d_group_members, // particles in active group
				unsigned int group_size, // number of particles in active group
				const BoxDim& box, // simulation box
				unsigned int block_size, // number of threads to use per block
				Scalar *d_charge, // pointer to particle charges
				Scalar *d_conductivity, // pointer to particle conductivities
				Scalar3 *d_dipole, // particle dipoles
				Scalar3 extfield, // external field
				Scalar3 *d_Eq, // external field less the charge contribution to the field 
				Scalar xi, // Ewald splitting parameter
				Scalar3 eta, // spectral splitting parameter
				Scalar rc, // real space cutoff radius
				Scalar drtable, // real space coefficient table spacing
				int Ntable, // number of entries in the real space coefficient table
				Scalar2 *d_phiS_table, // pointer to potential/dipole real space table
				Scalar4 *d_ES_table, // pointer to field/dipole real space table
				Scalar3 *d_gridk, // grid wave vectors
				Scalar *d_scale_phiS, // potential/dipole wave space scalings
				Scalar *d_scale_ES, // field/dipole wave space scalings
				CUFFTCOMPLEX *d_qgrid, // charge grid
				CUFFTCOMPLEX *d_SgridX, // x component of dipole grid
				CUFFTCOMPLEX *d_SgridY, // y component of dipole grid
				CUFFTCOMPLEX *d_SgridZ, // z component of dipole grid
				cufftHandle plan, // plan for the FFTs
				const int Nx, // number of grid nodes in the x dimension
				const int Ny, // number of grid nodes in the y dimension
				const int Nz, // number of grid nodes in the z dimension
				const unsigned int *d_n_neigh, // number of neighbors of each particle
				const unsigned int *d_nlist, // neighbor list
				const unsigned int *d_head_list, // used to access entries in the neighbor list
				int P, // number of grid nodes over which to spread and contract
				Scalar3 gridh, // grid spacing
				Scalar errortol) // error tolerance
{
	// Compute right side of M_ES * S = E0 - M_Eq * q
	FieldChargeMultiply(d_pos, d_group_membership, d_group_members, group_size, box, block_size, d_charge, d_Eq, extfield, xi, eta, rc, drtable, Ntable, d_phiS_table, d_gridk, d_scale_phiS, d_qgrid, d_SgridX, d_SgridY, d_SgridZ, plan, Nx, Ny, Nz, d_n_neigh, d_nlist, d_head_list, P, gridh); 

	// Create the matrix-free potential linear operator
	cuspPotential M(d_pos, d_group_membership, d_group_members, group_size, box, d_conductivity, xi, eta, rc, drtable, Ntable, d_ES_table, d_gridk, d_scale_ES, d_SgridX, d_SgridY, d_SgridZ, plan, Nx, Ny, Nz, d_n_neigh, d_nlist, d_head_list, gridh, P);

	// Allocate storage for the solution (S) and right side (rhs) on the GPU
	cusp::array1d<float, cusp::device_memory> S(M.num_rows, 0);
	cusp::array1d<float, cusp::device_memory> rhs(M.num_rows, 0);

	// Get pointers to the cusp arrays
	float *d_S = thrust::raw_pointer_cast(&S[0]);
	float *d_rhs = thrust::raw_pointer_cast(&rhs[0]);

	// Use the dipoles from the previous time step as the initial guess
	cudaMemcpy(d_S, d_dipole, 3*group_size*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_rhs, d_Eq, 3*group_size*sizeof(float), cudaMemcpyDeviceToDevice);

	// Set the preconditioner (identity for now)
	//cusp::identity_operator<float, cusp::device_memory> Pr(M.num_rows,M.num_rows);

	// Solve the linear system M_ES * S = E0 - M_Eq * q using GMRES
	cusp::default_monitor<float> monitor(rhs, 100, errortol);
	//cusp::krylov::cg(M, m, H, monitor);
	int restart = 10;
	cusp::krylov::gmres(M, S, rhs, restart, monitor);

	// Store the computed dipoles to the correct place in device memory
	cudaMemcpy(d_dipole, d_S, 3*group_size*sizeof(float), cudaMemcpyDeviceToDevice);

	// Print iteration number
	//if (monitor.converged())
        //{
        //    std::cout << "Solver converged after " << monitor.iteration_count() << " iterations." << std::endl;
        //}
        //else
        //{
        //    std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging." << std::endl;
        //}

    	return cudaSuccess;
}

cudaError_t gpu_ComputeForce(   Scalar4 *d_pos, // particle posisitons
				int *d_group_membership, // particle membership and index in active group
				unsigned int Ntotal, // total number of particles
				unsigned int *d_group_members, // particles in active group
				unsigned int group_size, // number of particles in active group
				const BoxDim& box, // simulation box
				unsigned int block_size, // number of threads to use per block
				Scalar4 *d_force, // pointer to particle forces
				Scalar *d_charge, // pointer to particle charges
				Scalar *d_conductivity, // pointer to particle conductivities
				Scalar3 *d_dipole, // particle dipoles
				Scalar3 extfield, // external field
				Scalar3 gradient, // external field gradient
				Scalar3 *d_Eq, // pointer to result of E0 - M_Eq * q
				Scalar xi, // Ewald splitting parameter
				Scalar3 eta, // spectral splitting parameter
				Scalar rc, // real space cutoff radius
				Scalar drtable, // real space coefficient table spacing
				int Ntable, // number of entries in the real space coefficient table
				Scalar2 *d_phiS_table, // pointer to potential/dipole real space table
				Scalar4 *d_ES_table, // pointer to field/dipole real space table
				Scalar2 *d_gradphiq_table, // pointer to potential/charge gradient real space table
				Scalar4 *d_gradphiS_table, // pointer to potential/dipole gradient real space table
				Scalar4 *d_gradES_table, // pointer to field/dipole gradient real space table
				Scalar3 *d_gridk, // grid wave vectors
				Scalar *d_scale_phiq, // potential/charge wave space scalings
				Scalar *d_scale_phiS, // potential/dipole wave space scalings
				Scalar *d_scale_ES, // field/dipole wave space scalings
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
				int dipoleflag) // indicates whether or not to turn off the mutual dipole functionality
{

	// total number of grid nodes
 	unsigned int Ngrid = Nx*Ny*Nz;

	// for grid initialization and scaling, use one thread per grid node
    	int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    	int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block per group particle.
	dim3 Nblocks2(group_size, 1, 1); // grid is a 1-D array
	dim3 Nthreads2(P, P, P); // block is a 3-D array

    	// for updating group membership and the real space calculation, use one thread per group particle
    	dim3 Nblocks3( (group_size/block_size) + 1, 1, 1);
    	dim3 Nthreads3(block_size, 1, 1);

	// for initializing group membership, use one thread per total particle
	dim3 Nblocks4( (Ntotal/block_size) + 1, 1, 1 );
	dim3 Nthreads4(block_size, 1, 1);

	// Factors needed for the kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xiterm = 2.0*xi*xi;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials

	// Handle the real space tables and positions as textured memory
    	phiS_table_tex.normalized = false; // Not normalized
    	phiS_table_tex.filterMode = cudaFilterModePoint; // Filter mode: floor of the index
    	// One dimension, Read mode: ElementType(Get what we write)
    	cudaBindTexture(0, phiS_table_tex, d_phiS_table, sizeof(Scalar2) * (Ntable+1));

    	ES_table_tex.normalized = false;
    	ES_table_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, ES_table_tex, d_ES_table, sizeof(Scalar4) * (Ntable+1));

    	gradphiq_table_tex.normalized = false;
    	gradphiq_table_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, gradphiq_table_tex, d_gradphiq_table, sizeof(Scalar2) * (Ntable+1));

    	gradphiS_table_tex.normalized = false;
    	gradphiS_table_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, gradphiS_table_tex, d_gradphiS_table, sizeof(Scalar4) * (Ntable+1));

	gradES_table_tex.normalized = false;
    	gradES_table_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, gradES_table_tex, d_gradES_table, sizeof(Scalar4) * (Ntable+1));

    	pos_tex.normalized = false;
    	pos_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, pos_tex, d_pos, sizeof(Scalar4) * Ntotal);

	// Update the group membership list
	initialize_groupmembership<<<Nblocks4, Nthreads4>>>(d_group_membership, Ntotal); // one thread per total particle
	groupmembership<<<Nblocks3, Nthreads3>>>(d_group_membership, d_group_members, group_size); 

	// Compute the particle dipoles. If constantdipoleflag = 1, this step is skipped and the particles keep their constant dipole model values that were precomputed on the host.
	if (dipoleflag != 1) {
		ComputeDipole(d_pos, d_group_membership, d_group_members, group_size, box, block_size, d_charge, 
			      d_conductivity, d_dipole, extfield, d_Eq, xi, eta, rc, drtable, Ntable, d_phiS_table, 
			      d_ES_table, d_gridk, d_scale_phiS, d_scale_ES, d_phiq_grid, d_ES_gridX, d_ES_gridY, 
			      d_ES_gridZ, plan, Nx, Ny, Nz, d_n_neigh, d_nlist, d_head_list, P, gridh, errortol);
	}

    	// Reset the grid values to zero
	initialize_grid<<<Nblocks1, Nthreads1>>>(d_phiq_grid,Ngrid);
	initialize_grid<<<Nblocks1, Nthreads1>>>(d_phiS_grid,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_Eq_gridX,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_Eq_gridY,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_Eq_gridZ,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_ES_gridX,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_ES_gridY,Ngrid);
    	initialize_grid<<<Nblocks1, Nthreads1>>>(d_ES_gridZ,Ngrid);

	// Spread charges and dipoles from the particles to the grid
	spread<<<Nblocks2, Nthreads2>>>(d_pos, d_charge, d_dipole, d_phiq_grid, d_ES_gridX, d_ES_gridY, d_ES_gridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, eta, xiterm, prefac);

	//  Compute the Fourier transform of the gridded data
	cufftExecC2C(plan, d_phiq_grid, d_phiq_grid, CUFFT_FORWARD);
    	cufftExecC2C(plan, d_ES_gridX, d_ES_gridX, CUFFT_FORWARD);
    	cufftExecC2C(plan, d_ES_gridY, d_ES_gridY, CUFFT_FORWARD);
    	cufftExecC2C(plan, d_ES_gridZ, d_ES_gridZ, CUFFT_FORWARD);

	// Scale the grid values
    	scale<<<Nblocks1, Nthreads1>>>(d_phiq_grid, d_phiS_grid, d_Eq_gridX, d_Eq_gridY, d_Eq_gridZ, d_ES_gridX, d_ES_gridY, d_ES_gridZ, d_gridk, d_scale_phiq, d_scale_phiS, d_scale_ES, Ngrid);

	// Inverse Fourier transform the gridded data
    	cufftExecC2C(plan, d_phiq_grid, d_phiq_grid, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_phiS_grid, d_phiS_grid, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_Eq_gridX, d_Eq_gridX, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_Eq_gridY, d_Eq_gridY, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_Eq_gridZ, d_Eq_gridZ, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_ES_gridX, d_ES_gridX, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_ES_gridY, d_ES_gridY, CUFFT_INVERSE);
    	cufftExecC2C(plan, d_ES_gridZ, d_ES_gridZ, CUFFT_INVERSE);

	// Contract the gridded values to the particles to get the wave space contribution to the force
	contract_force<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_pos, d_charge, d_dipole, d_force, d_phiq_grid, d_phiS_grid, d_Eq_gridX, d_Eq_gridY, d_Eq_gridZ, d_ES_gridX, d_ES_gridY, d_ES_gridZ, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, eta, xiterm, quadW*prefac);   

	// Compute the real space contribution to the force
    	real_space_force<<<Nblocks3, Nthreads3>>>(d_pos, d_charge, d_dipole, d_force, extfield, gradient, group_size, d_gradphiq_table, d_gradphiS_table, d_gradES_table, rc, Ntable, drtable, d_group_membership, d_group_members, box, d_n_neigh, d_nlist, d_head_list);

    	gpuErrchk(cudaPeekAtLastError());

	// Copy results to host (2 particles)
	//float4 *h_pos = (float4 *)malloc(2*sizeof(float4));
	//float *h_charge = (float *)malloc(2*sizeof(float));
	//float3 *h_dipole = (float3 *)malloc(2*sizeof(float3));
	//float4 *h_force = (float4 *)malloc(2*sizeof(float4));
	//cudaMemcpy(h_pos, d_pos, 2*sizeof(float4), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_charge, d_charge, 2*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_dipole, d_dipole, 2*sizeof(float3), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_force, d_force, 2*sizeof(float4), cudaMemcpyDeviceToHost);

	// Copy results to host (4 particles)
	//float4 *h_pos = (float4 *)malloc(4*sizeof(float4));
	//float *h_charge = (float *)malloc(4*sizeof(float));
	//float3 *h_dipole = (float3 *)malloc(4*sizeof(float3));
	//float4 *h_force = (float4 *)malloc(4*sizeof(float4));
	//cudaMemcpy(h_pos, d_pos, 4*sizeof(float4), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_charge, d_charge, 4*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_dipole, d_dipole, 4*sizeof(float3), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_force, d_force, 4*sizeof(float4), cudaMemcpyDeviceToHost);

	// Copy results to host (1 particle)
	//float4 *h_pos = (float4 *)malloc(sizeof(float4));
	//float *h_charge = (float *)malloc(sizeof(float));
	//float3 *h_dipole = (float3 *)malloc(sizeof(float3));
	//float4 *h_force = (float4 *)malloc(sizeof(float4));
	//cudaMemcpy(h_pos, d_pos, sizeof(float4), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_charge, d_charge, sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_dipole, d_dipole, sizeof(float3), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_force, d_force, sizeof(float4), cudaMemcpyDeviceToHost);

	// Display results (2 particles)
	//printf("Position: (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)\n", h_pos[0].x, h_pos[0].y, h_pos[0].z, h_pos[1].x, h_pos[1].y, h_pos[1].z);
	//printf("Charge: %.1f, %.1f\n", h_charge[0], h_charge[1]);
	//printf("Dipole: (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f)\n", h_dipole[0].x, h_dipole[0].y, h_dipole[0].z, h_dipole[1].x, h_dipole[1].y, h_dipole[1].z);
	//printf("Force: (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f)\n\n", h_force[0].x, h_force[0].y, h_force[0].z, h_force[1].x, h_force[1].y, h_force[1].z);

	// Display results (4 particles)
	//printf("Position: (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)\n", h_pos[0].x, h_pos[0].y, h_pos[0].z, h_pos[1].x, h_pos[1].y, h_pos[1].z, h_pos[2].x, h_pos[2].y, h_pos[2].z, h_pos[3].x, h_pos[3].y, h_pos[3].z);
	//printf("Charge: %.3f, %.3f, %.3f, %.3f\n", h_charge[0], h_charge[1], h_charge[2], h_charge[3]);
	//printf("Dipole: (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f)\n", h_dipole[0].x, h_dipole[0].y, h_dipole[0].z, h_dipole[1].x, h_dipole[1].y, h_dipole[1].z, h_dipole[2].x, h_dipole[2].y, h_dipole[2].z, h_dipole[3].x, h_dipole[3].y, h_dipole[3].z);
	//printf("Force: (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f)\n\n", h_force[0].x, h_force[0].y, h_force[0].z, h_force[1].x, h_force[1].y, h_force[1].z, h_force[2].x, h_force[2].y, h_force[2].z, h_force[3].x, h_force[3].y, h_force[3].z);

	// Free host memory
	//free(h_pos);
	//free(h_charge);
	//free(h_dipole);
	//free(h_force);

	cudaUnbindTexture(phiS_table_tex);
	cudaUnbindTexture(ES_table_tex);
	cudaUnbindTexture(gradphiq_table_tex);
	cudaUnbindTexture(gradphiS_table_tex);
    	cudaUnbindTexture(gradES_table_tex);
    	cudaUnbindTexture(pos_tex);

    	return cudaSuccess;
}

cudaError_t gpu_ComputeForce_Charge(    Scalar4 *d_pos, // particle posisitons
					int *d_group_membership, // particle membership and index in active group
					unsigned int Ntotal, // total number of particles
					unsigned int *d_group_members, // particles in active group
					unsigned int group_size, // number of particles in active group
					const BoxDim& box, // simulation box
					unsigned int block_size, // number of threads to use per block
					Scalar4 *d_force, // pointer to particle forces
					Scalar *d_charge, // pointer to particle charges
					Scalar3 extfield, // external field
					Scalar xi, // Ewald splitting parameter
					Scalar3 eta, // spectral splitting parameter
					Scalar rc, // real space cutoff radius
					Scalar drtable, // real space coefficient table spacing
					int Ntable, // number of entries in the real space coefficient table
					Scalar2 *d_gradphiq_table, // pointer to potential/charge gradient real space table
					Scalar *d_scale_phiq, // potential/charge wave space scalings
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
					Scalar errortol)
{

	// total number of grid nodes
 	unsigned int Ngrid = Nx*Ny*Nz;

	// for grid initialization and scaling, use one thread per grid node
    	int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    	int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block per group particle.
	dim3 Nblocks2(group_size, 1, 1); // grid is a 1-D array
	dim3 Nthreads2(P, P, P); // block is a 3-D array

    	// for updating group membership and the real space calculation, use one thread per group particle
    	dim3 Nblocks3( (group_size/block_size) + 1, 1, 1);
    	dim3 Nthreads3(block_size, 1, 1);

	// for initializing group membership, use one thread per total particle
	dim3 Nblocks4( (Ntotal/block_size) + 1, 1, 1 );
	dim3 Nthreads4(block_size, 1, 1);

	// Factors needed for the kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xiterm = 2.0*xi*xi;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials

	// Handle the real space tables and positions as textured memory
    	gradphiq_table_tex.normalized = false;
    	gradphiq_table_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, gradphiq_table_tex, d_gradphiq_table, sizeof(Scalar2) * (Ntable+1));

    	pos_tex.normalized = false;
    	pos_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, pos_tex, d_pos, sizeof(Scalar4) * Ntotal);

	// Update the group membership list
	initialize_groupmembership<<<Nblocks4, Nthreads4>>>(d_group_membership, Ntotal); // one thread per total particle
	groupmembership<<<Nblocks3, Nthreads3>>>(d_group_membership, d_group_members, group_size); 

    	// Reset the grid values to zero
	initialize_grid<<<Nblocks1, Nthreads1>>>(d_phiq_grid,Ngrid);

	// Spread charges from the particles to the grid
	spread_charge<<<Nblocks2, Nthreads2>>>(d_pos, d_charge, d_phiq_grid, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, eta, xiterm, prefac);

	//  Compute the Fourier transform of the gridded data
	cufftExecC2C(plan, d_phiq_grid, d_phiq_grid, CUFFT_FORWARD);

	// Scale the grid values
    	scale_chargepot<<<Nblocks1, Nthreads1>>>(d_phiq_grid, d_scale_phiq, Ngrid);

	// Inverse Fourier transform the gridded data
    	cufftExecC2C(plan, d_phiq_grid, d_phiq_grid, CUFFT_INVERSE);

	// Contract the gridded values to the particles to get the wave space contribution to the force
	contract_force_charge<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_pos, d_charge, d_force, d_phiq_grid, group_size, Nx, Ny, Nz, d_group_members, box, P, gridh, eta, xiterm, quadW*prefac);   

	// Compute the real space contribution to the force
    	real_space_force_charge<<<Nblocks3, Nthreads3>>>(d_pos, d_charge, d_force, extfield, group_size, d_gradphiq_table, rc, Ntable, drtable, d_group_membership, d_group_members, box, d_n_neigh, d_nlist, d_head_list);

    	gpuErrchk(cudaPeekAtLastError());

	cudaUnbindTexture(phiS_table_tex);
    	cudaUnbindTexture(pos_tex);

    	return cudaSuccess;
}

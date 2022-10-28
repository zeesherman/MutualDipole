#include "Dielectric2.cuh"
#include <stdio.h>

#include <cusp/linear_operator.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

// Construct class wrapper to use the grand potential matrix as a matrix-free method in CUSP.
class cuspPotential : public cusp::linear_operator<float,cusp::device_memory>
{
public:

    typedef cusp::linear_operator<float,cusp::device_memory> super; // Defines size of linear operator
  
    unsigned int group_size; // Number of particles

    Scalar xi;  // Ewald splitting parameter
    Scalar3 eta;  // Spectral splitting parameter
    Scalar rc;  // cut-off distance for real space calculations
    Scalar drtable;  // real space table discretization
    int Ntable;  // number of entries in real space table

    const BoxDim& box;  // size of the simulation box
    const int Nx;  // number of grid nodes in x direction
    const int Ny;  // number of grid nodes in y direction
    const int Nz;  // number of grid nodes in z direction
    Scalar3 gridh;  // grid spacing
    int P;  // number of grid nodes in Gaussian support
    cufftHandle plan;  // plan for cuFFT
 
    Scalar4 *d_pos;  // particle positions and types
    Scalar *d_conductivity; // particle conductivity
    int *d_group_membership; // particle membership and index in group for which the force calculation is being performed
    unsigned int *d_group_members;  // index into particle tag
    const unsigned int *d_n_neigh;  // number of neighbors of each particle
    const unsigned int *d_nlist;    // neighbor list
    const unsigned int *d_head_list;  // used to access entries in the neighbor list

    Scalar4 *d_ES_table;  // field/dipole real space field table

    Scalar3 *d_gridk;  // wave vectors corresponding to grid points
    Scalar *d_scale_ES; // field/dipole scaling at each grid point
    CUFFTCOMPLEX *d_SgridX;  // x-component of dipole grid
    CUFFTCOMPLEX *d_SgridY;  // y-component of dipole grid
    CUFFTCOMPLEX *d_SgridZ;  // z-component of dipole grid

    // constructor
    cuspPotential(Scalar4 *d_pos,
		  int *d_group_membership,
		  unsigned int *d_group_members,
		  unsigned int group_size,
		  const BoxDim& box,
		  Scalar *d_conductivity,
		  Scalar xi,
                  Scalar3 eta,
                  Scalar rc,
                  Scalar drtable,
                  int Ntable,
		  Scalar4 *d_ES_table,
		  Scalar3 *d_gridk,
		  Scalar *d_scale_ES,
                  CUFFTCOMPLEX *d_SgridX,
                  CUFFTCOMPLEX *d_SgridY,
                  CUFFTCOMPLEX *d_SgridZ,
		  cufftHandle plan,
                  const int Nx,
                  const int Ny,
                  const int Nz,
		  const unsigned int *d_n_neigh,
                  const unsigned int *d_nlist,
		  const unsigned int *d_head_list,
                  Scalar3 gridh,
                  int P)
                  : super(3*group_size,3*group_size), 
		  d_pos(d_pos),
		  d_group_membership(d_group_membership),
		  d_group_members(d_group_members),
    		  group_size(group_size),
		  box(box),
		  d_conductivity(d_conductivity),
		  xi(xi),
                  eta(eta),
                  rc(rc),
                  drtable(drtable),
                  Ntable(Ntable),
		  d_ES_table(d_ES_table),
		  d_gridk(d_gridk),
		  d_scale_ES(d_scale_ES),
		  d_SgridX(d_SgridX),
		  d_SgridY(d_SgridY),
		  d_SgridZ(d_SgridZ),
                  plan(plan),
                  Nx(Nx),
                  Ny(Ny),
                  Nz(Nz),
		  d_n_neigh(d_n_neigh),
		  d_nlist(d_nlist),
		  d_head_list(d_head_list),
                  gridh(gridh),
                  P(P){}


    // linear operator y = A*x
    //! Matrix multiplication part of CUSP wrapper
    template <typename VectorType1,
             typename VectorType2>
    void operator()( VectorType1& x, VectorType2& y )
    {

        // obtain a raw pointer to device memory for input and output
        // arrays
        float *x_ptr = (float *) thrust::raw_pointer_cast(&x[0]);
        float *y_ptr = (float *) thrust::raw_pointer_cast(&y[0]);

	// Cast to a Scalar3 pointer
	Scalar3 *x_ptr2 = (Scalar3 *) &x_ptr[0];
	Scalar3 *y_ptr2 = (Scalar3 *) &y_ptr[0];

        // run kernels to compute y = A*x
	FieldDipoleMultiply(	d_pos,
			d_group_membership,
			d_group_members,
			group_size,
			box,
			512, // blocksize
			x_ptr2,
			d_conductivity,
			y_ptr2,
			xi,
			eta,
			rc,
			drtable,
			Ntable,
			d_ES_table,
			d_gridk,
			d_scale_ES,
			d_SgridX,
			d_SgridY,
			d_SgridZ,
			plan,
			Nx,
			Ny,
			Nz,
			d_n_neigh,
			d_nlist,
			d_head_list,
			P,
			gridh);

    }
};


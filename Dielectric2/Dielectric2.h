#include <hoomd/ForceCompute.h>
#include <cufft.h>
#include <hoomd/md/NeighborList.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>

#ifndef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

#ifndef __DIELECTRIC2_H__
#define __DIELECTRIC2_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

// Declares the Dielectric2 class.
class Dielectric2 : public ForceCompute {

    public:
        // Constructs the compute and associates it with the system
        Dielectric2(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
	            std::shared_ptr<NeighborList> nlist,
		    std::vector<float> &field,
		    std::vector<float> &gradient,
		    std::vector<float> &conductivity,
	            Scalar xi,
		    Scalar errortol,
		    std::string fileprefix,
		    int period,
		    int dipoleflag,
		    unsigned int t0);

	// Destructor
        virtual ~Dielectric2();

	// Set parameters needed for the force calculation
	void SetParams();

	// Update the external field
	void UpdateField(std::vector<float> &field,
			 std::vector<float> &gradient);

	// Update simulation parameters
	void UpdateParameters(std::vector<float> &field,
			      std::vector<float> &gradient,
			      std::vector<float> &conductivity,
			      std::string fileprefix,
			      int period,
			      int dipoleflag,
			      unsigned int t0);

	// Performs the force calculation
        virtual void computeForces(unsigned int timestep);

	// Write particle positions, dipoles, and forces to file
	void OutputData(unsigned int timestep);

    protected:

	std::shared_ptr<ParticleGroup> m_group;		// active group of particles on which to perform the calculation
	Scalar3 m_field;				// applied external field
	Scalar3 m_gradient;				// applied field gradient
	GPUArray<Scalar3> m_dipole;			// particle dipoles
	GPUArray<Scalar> m_conductivity;		// particle conductivities

	int m_Ntotal;					// total number of particles
	int m_group_size;				// number of particles in the active group
	GPUArray<int> m_group_membership;		// active group membership list

	Scalar m_xi;               			// Ewald splitting parameter
	Scalar m_errortol;				// error tolerance
	Scalar3 m_eta;					// spectral splitting parameter
	Scalar m_rc;                           		// real space cutoff
	int m_Nx;                                	// number of grid points in x direction
	int m_Ny;                                	// number of grid points in y direction
	int m_Nz;                                	// number of grid points in z direction
	Scalar3 m_gridh;				// grid spacing
	int m_P;					// number of grid points over which to spread and contract

	GPUArray<Scalar3> m_gridk;      		// wave vector corresponding to each grid point
	GPUArray<Scalar> m_scale_phiq;			// potential/charge scaling on grid
	GPUArray<Scalar> m_scale_phiS;			// potential/dipole scaling on grid
	GPUArray<Scalar> m_scale_ES;			// field/dipole scaling on grid
	GPUArray<CUFFTCOMPLEX> m_phiq_grid;      	// potential/charge grid
	GPUArray<CUFFTCOMPLEX> m_phiS_grid;      	// potential/dipole grid
	GPUArray<CUFFTCOMPLEX> m_Eq_gridX;      	// x component of field/charge grid
	GPUArray<CUFFTCOMPLEX> m_Eq_gridY;      	// y component of field/charge grid
	GPUArray<CUFFTCOMPLEX> m_Eq_gridZ;      	// z component of field/charge grid
	GPUArray<CUFFTCOMPLEX> m_ES_gridX;      	// x component of field/dipole grid
	GPUArray<CUFFTCOMPLEX> m_ES_gridY;      	// y component of field/dipole grid
	GPUArray<CUFFTCOMPLEX> m_ES_gridZ;      	// z component of field/dipole grid
	GPUArray<Scalar3> m_Eq;				// result of E0 - M_Eq * q, the external field less the charge contribution to the field
	cufftHandle m_plan;                    		// used for the fast Fourier transformations performed on the GPU

	int m_Ntable;                           	// number of entries in real space tables
	Scalar m_drtable;                           	// real space table spacing
	GPUArray<Scalar2> m_phiS_table;			// potential/dipole real space table
	GPUArray<Scalar4> m_ES_table;                   // field/dipole real space table
	GPUArray<Scalar2> m_gradphiq_table;		// potential/charge gradient real space table
	GPUArray<Scalar4> m_gradphiS_table;		// potential/dipole gradient real space table
	GPUArray<Scalar4> m_gradES_table;		// field/dipole gradient real space table

	std::shared_ptr<NeighborList> m_nlist;    	// neighbor list

	std::string m_fileprefix;			// output file prefix
	int m_period;					// frequency with which to write output files
	unsigned int m_t0;				// initial timestep

	int m_dipoleflag;				// 0 indicates mutual dipole model (particles can mutually polarize each other)
							// 1 indicates constant dipole model (particles are only polarized by external field)
							// 2 indicates no polarization (charge/charge interactions only)

    };

// Exports the Dielectric2 class to python
void export_Dielectric2(pybind11::module& m);

#endif

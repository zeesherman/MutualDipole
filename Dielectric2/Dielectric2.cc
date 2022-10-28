#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "Dielectric2.h"
#include "Dielectric2.cuh"
#include "PotentialWrapper.cuh"
#include <stdio.h>
#include <algorithm>
#include <iomanip>
#include <cmath>

#define PI 3.1415926535897932

using namespace std;

// Constructor for the Dielectric2 class
Dielectric2::Dielectric2( std::shared_ptr<SystemDefinition> sysdef, // system this method will act on; must not be NULL
			  std::shared_ptr<ParticleGroup> group, // group of particles for which to compute the force
			  std::shared_ptr<NeighborList> nlist, // neighbor list
			  std::vector<float> &conductivity, // particle conductivities
			  std::vector<float> &field, // imposed external field
			  std::vector<float> &gradient, // imposed external field gradient
			  Scalar xi, // Ewald splitting parameter
			  Scalar errortol, // error tolerance
			  std::string fileprefix,  // output file name prefix
			  int period,  // output file period
			  int dipoleflag,  // indicates whether to turn off the dipole functionality
			  unsigned int t0) // initial timestep
			  : ForceCompute(sysdef),
			  m_group(group),
			  m_nlist(nlist),
			  m_xi(xi),
			  m_errortol(errortol),
			  m_fileprefix(fileprefix),
			  m_period(period),
			  m_dipoleflag(dipoleflag),
			  m_t0(t0)
{
    m_exec_conf->msg->notice(5) << "Constructing Dielectric2" << std::endl;

	// only one GPU is supported
	if (!m_exec_conf->isCUDAEnabled())
	{
		m_exec_conf->msg->error() << "Creating a Dielectric2 when CUDA is disabled" << std::endl;
		throw std::runtime_error("Error initializing Dielectric2");
	}

	// Set the field and field gradient
	m_field = make_scalar3(field[0], field[1], field[2]);
	m_gradient = make_scalar3(gradient[0], gradient[1], gradient[2]);

	// Get the group size and total number of particles
	m_group_size = m_group->getNumMembers();
	m_Ntotal = m_pdata->getN();
	
	// Extract the particle conductivities
	GPUArray<Scalar> n_conductivity(m_group_size, m_exec_conf);
	m_conductivity.swap(n_conductivity);
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::readwrite);
	for (unsigned int i = 0; i < m_group_size; ++i ){
		h_conductivity.data[i] = conductivity[i];
	}

}

// Destructor for the Dielectric2 class
Dielectric2::~Dielectric2() {

    m_exec_conf->msg->notice(5) << "Destroying Dielectric2" << std::endl;
	cufftDestroy(m_plan);
}

// Compute and set parameters needed for the calculations.  This step is computed only once when the
// Dielectric2 class is created or on each call to update_parameters if system parameters change.
void Dielectric2::SetParams() {

	////// Compute parameters associated with the numerical method.

	const BoxDim& box = m_pdata->getBox(); // simulation box
	Scalar3 L = box.getL();  // box dimensions

	m_rc = sqrtf(-logf(m_errortol))/m_xi;  // real space cutoff radius
	Scalar kcut = 2.0*m_xi*sqrtf(-logf(m_errortol));  // wave space cutoff
	m_Nx = int(ceil(1.0 + L.x*kcut/PI));  // number of grid nodes in the x direction
	m_Ny = int(ceil(1.0 + L.y*kcut/PI));  // number of grid nodes in the y direction
	m_Nz = int(ceil(1.0 + L.z*kcut/PI));  // number of grid nodes in the z direction

	// Get a list of 5-smooth integers between 8 and 512^3 (can be written as (2^a)*(3^b)*(5^c); i.e. only prime factors of 2, 3, and 5)
	std::vector<int> Mlist;
	for ( int ii = 0; ii < 28; ++ii ){
		int pow2 = 1;
		for ( int i = 0; i < ii; ++i ){
			pow2 *= 2;
		}
		for ( int jj = 0; jj < 18; ++jj ){
			int pow3 = 1;
			for ( int j = 0; j < jj; ++j ){
				pow3 *= 3;
			}
			for ( int kk = 0; kk < 12; ++kk ){
				int pow5 = 1;
				for ( int k = 0; k < kk; ++k ){
					pow5 *= 5;
				}
				int Mcurr = pow2 * pow3 * pow5;
				if ( Mcurr >= 8 && Mcurr <= 512*512*512 ){
					Mlist.push_back(Mcurr);
				}
			}
		}
	}

	// Sort the list from lowest to highest
	std::sort(Mlist.begin(), Mlist.end());

	// Get the length of the list
	const int nmult = Mlist.size();		

	// Set the number of grid points to be a 5-smooth integer for most efficient FFTs
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nx <= Mlist[ii]){
			m_Nx = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Ny <= Mlist[ii]){
			m_Ny = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nz <= Mlist[ii]){
			m_Nz = Mlist[ii];
			break;
		}
	}

	// Throw an error if there are too many grid nodes
	if (m_Nx*m_Ny*m_Nz > 512*512*512){
		printf("ERROR: Total number of grid nodes is larger than 512^3.  Calculation may fail due to GPU memory limitations. Try decreasing the error tolerance, decreasing the Ewald splitting parameter, or decreasing the size of the simulation box.\n");
		printf("Nx = %i \n", m_Nx);
		printf("Ny = %i \n", m_Ny);
		printf("Nz = %i \n", m_Nz);
		printf("Total = %i \n", m_Nx*m_Ny*m_Nz);

		exit(EXIT_FAILURE);
	}

	// Additional parameters
	Scalar Ngrid = m_Nx*m_Ny*m_Nz; // total number of grid nodes
	m_gridh = L / make_scalar3(m_Nx,m_Ny,m_Nz); // grid spacing
	m_P = ceil(-2.0*logf(m_errortol)/PI);  // number of grid nodes over which to support spreading and contracting kernels
	m_eta = m_P*m_gridh*m_gridh*m_xi*m_xi/PI; // spectral splitting parameter controlling decay of spreading and contracting kernels

	// Cannot support more nodes than the grid size
	if (m_P > m_Nx) m_P = m_Nx;
	if (m_P > m_Ny) m_P = m_Ny;		     
	if (m_P > m_Nz) m_P = m_Nz;

	// Print summary to command line output
	printf("\n");
	printf("\n");
	m_exec_conf->msg->notice(2) << "--- Parameters ---" << std::endl;
	m_exec_conf->msg->notice(2) << "Active group size: " << m_group_size << std::endl;
	m_exec_conf->msg->notice(2) << "Box dimensions: " << L.x << ", " << L.y << ", " << L.z << std::endl;
	m_exec_conf->msg->notice(2) << "Ewald parameter xi: " << m_xi << std::endl;
	m_exec_conf->msg->notice(2) << "Error tolerance: " << m_errortol << std::endl;
	m_exec_conf->msg->notice(2) << "Real space cutoff: " << m_rc << std::endl;
	m_exec_conf->msg->notice(2) << "Wave space cutoff: " << kcut << std::endl;
	m_exec_conf->msg->notice(2) << "Grid nodes in each dimension: " << m_Nx << ", " << m_Ny << ", " << m_Nz << std::endl;
	m_exec_conf->msg->notice(2) << "Grid spacing: " << m_gridh.x << ", " << m_gridh.y << ", " << m_gridh.z << std::endl;
	m_exec_conf->msg->notice(2) << "Spectral parameter eta: " << m_eta.x << ", " << m_eta.y << ", " << m_eta.z << std::endl;
	m_exec_conf->msg->notice(2) << "Support P: " << m_P << std::endl;
	printf("\n");
	printf("\n");

	////// Wave space grid

	// Create plan for FFTs on the GPU
	cufftPlan3d(&m_plan, m_Nx, m_Ny, m_Nz, CUFFT_C2C);

	// Initialize array for the wave vector at each grid point
	GPUArray<Scalar3> n_gridk(Ngrid, m_exec_conf);
	m_gridk.swap(n_gridk);
	ArrayHandle<Scalar3> h_gridk(m_gridk, access_location::host, access_mode::readwrite);

	// Initialize arrays for the Wave space scalings
	GPUArray<Scalar> n_scale_phiq(Ngrid, m_exec_conf);
	GPUArray<Scalar> n_scale_phiS(Ngrid, m_exec_conf);
	GPUArray<Scalar> n_scale_ES(Ngrid, m_exec_conf);
	m_scale_phiq.swap(n_scale_phiq);
	m_scale_phiS.swap(n_scale_phiS);
	m_scale_ES.swap(n_scale_ES);
	ArrayHandle<Scalar> h_scale_phiq(m_scale_phiq, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar> h_scale_phiS(m_scale_phiS, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar> h_scale_ES(m_scale_ES, access_location::host, access_mode::readwrite);

	// Grid arrays
	GPUArray<CUFFTCOMPLEX> n_phiq_grid(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_phiS_grid(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_Eq_gridX(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_Eq_gridY(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_Eq_gridZ(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_ES_gridX(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_ES_gridY(Ngrid, m_exec_conf);
	GPUArray<CUFFTCOMPLEX> n_ES_gridZ(Ngrid, m_exec_conf);
	m_phiq_grid.swap(n_phiq_grid);
	m_phiS_grid.swap(n_phiS_grid);
	m_Eq_gridX.swap(n_Eq_gridX);
	m_Eq_gridY.swap(n_Eq_gridY);
	m_Eq_gridZ.swap(n_Eq_gridZ);
	m_ES_gridX.swap(n_ES_gridX);
	m_ES_gridY.swap(n_ES_gridY);
	m_ES_gridZ.swap(n_ES_gridZ);

	// Populate grids with wave space vectors and scalings
	for (int i = 0; i < m_Nx; i++) {  // loop through x dimension
		for (int j = 0; j < m_Ny; j++) {  // loop through y dimension
			for (int k = 0; k < m_Nz; k++) {  // loop through z dimension

				// Linear index into grid array
				int idx = i*m_Ny*m_Nz + j*m_Nz + k;

				// wave vector components goes from -2*PI*N/2 to 2*PI*N/2
				h_gridk.data[idx].x = ((i < (m_Nx+1)/2) ? i : i - m_Nx) * 2.0*PI/L.x;
				h_gridk.data[idx].y = ((j < (m_Ny+1)/2) ? j : j - m_Ny) * 2.0*PI/L.y;
				h_gridk.data[idx].z = ((k < (m_Nz+1)/2) ? k : k - m_Nz) * 2.0*PI/L.z;

				// wave vector magnitude and magnitude squared
				Scalar k2 = h_gridk.data[idx].x*h_gridk.data[idx].x + h_gridk.data[idx].y*h_gridk.data[idx].y + h_gridk.data[idx].z*h_gridk.data[idx].z;
				Scalar kmag = sqrt(k2);

				// term in the exponential of the scaling factors
				Scalar etak2 = (1.0-m_eta.x)*h_gridk.data[idx].x*h_gridk.data[idx].x + (1.0-m_eta.y)*h_gridk.data[idx].y*h_gridk.data[idx].y + (1.0-m_eta.z)*h_gridk.data[idx].z*h_gridk.data[idx].z;

				// Scaling factor used in wave space sum.  k = 0 term is excluded.
				if (i == 0 && j == 0 && k == 0){
					h_scale_phiq.data[idx] = 0.0;
					h_scale_phiS.data[idx] = 0.0;
					h_scale_ES.data[idx] = 0.0;
				}
				else{
					// Divided by extra factors of k to change the k vectors in the scaling kernel to unit vectors.  Divided by total number of grid nodes due to the ifft conventions in cuFFT.
					h_scale_phiq.data[idx] = pow(sin(kmag)/kmag,2)*expf(-etak2/(4.0*m_xi*m_xi))/k2 / Ngrid;
					h_scale_phiS.data[idx] = -3.0*sin(kmag)/kmag*(sin(kmag)/k2 - cos(kmag)/kmag)*expf(-etak2/(4.0*m_xi*m_xi))/(k2*kmag) / Ngrid;
					h_scale_ES.data[idx] = 9.0*pow( (sin(kmag)/k2 - cos(kmag)/kmag) ,2)*expf(-etak2/(4.0*m_xi*m_xi))/(k2*k2) / Ngrid;
				} // end scaling if statement
				
			} // end z component loop (k)
		} // end y component loop (j)
	} // end x component loop (i)

	////// Real space tables

	// Parameters for the real space table
	m_drtable = double(0.001); // table spacing
	m_Ntable = m_rc/m_drtable - 1; // number of entries in the table

	// Allocate a GPUArray and Array handle for the real space tables
	//
	// Potential/dipole or field/charge real space table
	GPUArray<Scalar2> n_phiS_table((m_Ntable+1), m_exec_conf);
	m_phiS_table.swap(n_phiS_table);
	ArrayHandle<Scalar2> h_phiS_table(m_phiS_table, access_location::host, access_mode::readwrite);

	// Field/dipole real space table
	GPUArray<Scalar4> n_ES_table((m_Ntable+1), m_exec_conf);
	m_ES_table.swap(n_ES_table);
	ArrayHandle<Scalar4> h_ES_table(m_ES_table, access_location::host, access_mode::readwrite);

	// Potential/charge gradient real space table
	GPUArray<Scalar2> n_gradphiq_table((m_Ntable+1), m_exec_conf);
	m_gradphiq_table.swap(n_gradphiq_table);
	ArrayHandle<Scalar2> h_gradphiq_table(m_gradphiq_table, access_location::host, access_mode::readwrite);

	// Potential/dipole or field/charge gradient real space table
	GPUArray<Scalar4> n_gradphiS_table((m_Ntable+1), m_exec_conf);
	m_gradphiS_table.swap(n_gradphiS_table);
	ArrayHandle<Scalar4> h_gradphiS_table(m_gradphiS_table, access_location::host, access_mode::readwrite);

	// Field/dipole gradient real space table
	GPUArray<Scalar4> n_gradES_table((m_Ntable+1), m_exec_conf);
	m_gradES_table.swap(n_gradES_table);
	ArrayHandle<Scalar4> h_gradES_table(m_gradES_table, access_location::host, access_mode::readwrite);

	// xi values
	double xi = m_xi;
	double xi2 = pow(xi,2);
	double xi3 = pow(xi,3);
	double xi4 = pow(xi,4);
	double xi5 = pow(xi,5);
	double xi6 = pow(xi,6);

	// Fill the real space tables
	for (int i = 0; i <= m_Ntable; i++)
	{

		// Particle separation corresponding to current table entry
		double dist = (i + 1) * m_drtable;		
		double dist2 = pow(dist,2);
		double dist3 = pow(dist,3);
		double dist4 = pow(dist,4);
		double dist5 = pow(dist,5);
		double dist6 = pow(dist,6);

		// exponentials and complimentary error functions
		double expp = exp(-(dist+2)*(dist+2)*xi2);		
		double expm = exp(-(dist-2)*(dist-2)*xi2);
		double exp0 = exp(-dist2*xi2);
		double erfp = erfc((dist+2)*xi);
		double erfm = erfc((dist-2)*xi);
		double erf0 = erfc(dist*xi);

		// Potential/dipole or field/charge table
		double exppolyp = 1./(256.*pow(PI,1.5)*xi3*dist2)*(-6.*xi2*dist3 - 4.*xi2*dist2 + (-3.+8.*xi2)*dist + 2.*(1.-8.*xi2));
		double exppolym = 1./(256.*pow(PI,1.5)*xi3*dist2)*(-6.*xi2*dist3 + 4.*xi2*dist2 + (-3.+8.*xi2)*dist - 2.*(1.-8.*xi2));
		double exppoly0 = 3.*(2.*dist2*xi2+1.)/(128.*pow(PI,1.5)*xi3*dist);
		double erfpolyp = 1./(512*PI*xi4*dist2)*(12.*xi4*dist4 + 32.*xi4*dist3 + 12.*xi2*dist2 - 3.+64.*xi4);
		double erfpolym = 1./(512*PI*xi4*dist2)*(12.*xi4*dist4 - 32.*xi4*dist3 + 12.*xi2*dist2 - 3.+64.*xi4);
		double erfpoly0 = -3.*(4.*xi4*dist4 + 4.*xi2*dist2 - 1.)/(256*PI*xi4*dist2);

		// Regularization for overlapping particles
		double regpoly;
		if (dist < 2) {
			regpoly = -1./(4.*PI*dist2) + dist/(8.*PI)*(1. - 3.*dist/8.);
		} else {
			regpoly = 0.;
		}

		// Enter the table values
		h_phiS_table.data[i].x = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);

		// Field/dipole table

		// I-rr polynomials
		exppolyp = 1./(1024.*pow(PI,1.5)*xi5*dist3)*(4.*xi4*dist5 - 8.*xi4*dist4 + 8.*xi2*(2.-7.*xi2)*dist3 - 8.*xi2*(3.+2.*xi2)*dist2 + (3.-12.*xi2+32.*xi4)*dist + 2.*(3.+4.*xi2-32.*xi4));
		exppolym = 1./(1024.*pow(PI,1.5)*xi5*dist3)*(4.*xi4*dist5 + 8.*xi4*dist4 + 8.*xi2*(2.-7.*xi2)*dist3 + 8.*xi2*(3.+2.*xi2)*dist2 + (3.-12.*xi2+32.*xi4)*dist - 2.*(3.+4.*xi2-32.*xi4));
		exppoly0 = 1./(512.*pow(PI,1.5)*xi5*dist2)*(-4.*xi4*dist4 - 8.*xi2*(2.-9.*xi2)*dist2 - 3.+36.*xi2);
		erfpolyp = 1./(2048.*PI*xi6*dist3)*(-8.*xi6*dist6 - 36.*xi4*(1.-4.*xi2)*dist4 + 256.*xi6*dist3 - 18.*xi2*(1.-8.*xi2)*dist2 + 3.-36.*xi2+256.*xi6);
		erfpolym = 1./(2048.*PI*xi6*dist3)*(-8.*xi6*dist6 - 36.*xi4*(1.-4.*xi2)*dist4 - 256.*xi6*dist3 - 18.*xi2*(1.-8.*xi2)*dist2 + 3.-36.*xi2+256.*xi6);
		erfpoly0 = 1./(1024.*PI*xi6*dist3)*(8.*xi6*dist6 + 36.*xi4*(1.-4.*xi2)*dist4 + 18.*xi2*(1.-8.*xi2)*dist2 - 3.+36.*xi2);

		// Regularization for overlapping particles
		if (dist < 2) {
			regpoly =  -1./(4.*PI*dist3) + 1./(4.*PI)*(1. - 9.*dist/16. + dist3/32.);
		} else {
			regpoly = 0.;
		}
 
		// I-rr term gets the .x field
		h_ES_table.data[i].x = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);

		// rr polynomials 
		exppolyp = 1./(512.*pow(PI,1.5)*xi5*dist3)*(8.*xi4*dist5 - 16.*xi4*dist4 + 2.*xi2*(7.-20.*xi2)*dist3 - 4.*xi2*(3.-4.*xi2)*dist2 - (3.-12.*xi2+32.*xi4)*dist - 2.*(3.+4.*xi2-32.*xi4));
		exppolym = 1./(512.*pow(PI,1.5)*xi5*dist3)*(8.*xi4*dist5 + 16.*xi4*dist4 + 2.*xi2*(7.-20.*xi2)*dist3 + 4.*xi2*(3.-4.*xi2)*dist2 - (3.-12.*xi2+32.*xi4)*dist + 2.*(3.+4.*xi2-32.*xi4));
		exppoly0 = 1./(256.*pow(PI,1.5)*xi5*dist2)*(-8.*xi4*dist4 - 2.*xi2*(7.-36.*xi2)*dist2 + 3.-36.*xi2);
		erfpolyp = 1./(1024.*PI*xi6*dist3)*(-16.*xi6*dist6 - 36.*xi4*(1.-4.*xi2)*dist4 + 128.*xi6*dist3 - 3.+36.*xi2-256.*xi6);
		erfpolym = 1./(1024.*PI*xi6*dist3)*(-16.*xi6*dist6 - 36.*xi4*(1.-4.*xi2)*dist4 - 128.*xi6*dist3 - 3.+36.*xi2-256.*xi6);
		erfpoly0 = 1./(512.*PI*xi6*dist3)*(16.*xi6*dist6 + 36.*xi4*(1.-4.*xi2)*dist4 + 3.-36.*xi2);

		// Regularization for overlapping particles
		if (dist < 2) {
			regpoly =  1./(2.*PI*dist3) + 1./(4.*PI)*(1. - 9.*dist/8. + dist3/8.);
		} else {
			regpoly = 0.;
		}

		// rr term gets the .y field
		h_ES_table.data[i].y = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);

		// Potential/charge gradient table
		exppolyp = -(dist-2.)/(32.*pow(PI,1.5)*xi*dist2);
		exppolym = -(dist+2.)/(32.*pow(PI,1.5)*xi*dist2);
		exppoly0 = 1./(16.*pow(PI,1.5)*xi*dist);
		erfpolyp = (2.*xi2*dist2 - 8.*xi2 - 1.)/(64.*PI*xi2*dist2);
		erfpolym = (2.*xi2*dist2 - 8.*xi2 - 1.)/(64.*PI*xi2*dist2);
		erfpoly0 = -(2.*xi2*dist2 - 1.)/(32.*PI*xi2*dist2);

		// Regularization for overlapping particles
		if (dist < 2) {
			regpoly =  1./(4.*PI*dist2) - 1./(16.*PI);
		} else {
			regpoly = 0.;
		}

		// Enter the table values
		h_gradphiq_table.data[i].x = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);

		// Potential/dipole or field/charge gradient table

		// I-rr component gets the .x field
		h_gradphiS_table.data[i].x = h_phiS_table.data[i].x/dist;

		// rr component
		exppolyp = 1./(128.*pow(PI,1.5)*xi3*dist3)*(-6.*xi2*dist3 + 4.*xi2*dist2 + (3.-8.*xi2)*dist - 2.*(1.-8.*xi2));
		exppolym = 1./(128.*pow(PI,1.5)*xi3*dist3)*(-6.*xi2*dist3 - 4.*xi2*dist2 + (3.-8.*xi2)*dist + 2.*(1.-8.*xi2));
		exppoly0 = 3.*(2.*xi2*dist2 - 1.)/(64.*pow(PI,1.5)*xi3*dist2);
		erfpolyp = 1./(256.*PI*xi4*dist3)*(12.*xi4*dist4 + 16.*xi4*dist3 + 3.-64.*xi4);
		erfpolym = 1./(256.*PI*xi4*dist3)*(12.*xi4*dist4 - 16.*xi4*dist3 + 3.-64.*xi4);
		erfpoly0 = -3.*(4.*xi4*dist4 + 1.)/(128.*PI*xi4*dist3);

		// Regularization for overlapping particles
		if (dist < 2) {
			regpoly =  1./(2.*PI*dist3) + 1./(32.*PI)*(4.-3.*dist);
		} else {
			regpoly = 0.;
		}

		// rr component gets the .y field
		h_gradphiS_table.data[i].y = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);

		// Field/dipole gradient table

		// -( (Si*Sj)r + (Sj*r)Si + (Si*r)Sj - 2(Si*r)(Sj*r)r ) polynomials 
		exppolyp = 3./(1024.*pow(PI,1.5)*xi5*dist4)*(4.*xi4*dist5 - 8.*xi4*dist4 + 4.*xi2*(1.-2.*xi2)*dist3 + 16.*xi4*dist2 - (3.-12.*xi2+32.*xi4)*dist - 2.*(3.+4.*xi2-32.*xi4));
		exppolym = 3./(1024.*pow(PI,1.5)*xi5*dist4)*(4.*xi4*dist5 + 8.*xi4*dist4 + 4.*xi2*(1.-2.*xi2)*dist3 - 16.*xi4*dist2 - (3.-12.*xi2+32.*xi4)*dist + 2.*(3.+4.*xi2-32.*xi4));
		exppoly0 = 3./(512.*pow(PI,1.5)*xi5*dist3)*(-4.*xi4*dist4 - 4.*xi2*(1.-6.*xi2)*dist2 + 3.-36.*xi2);
		erfpolyp = 3./(2048.*PI*xi6*dist4)*(-8.*xi6*dist6 - 12.*xi4*(1.-4.*xi2)*dist4 + 6.*xi2*(1.-8.*xi2)*dist2 - 3.+36.*xi2-256.*xi6);
		erfpolym = erfpolyp;
		erfpoly0 = 3./(1024.*PI*xi6*dist4)*(8.*xi6*dist6 + 12.*xi4*(1.-4.*xi2)*dist4 - 6.*xi2*(1.-8.*xi2)*dist2 + 3.-36.*xi2);

		// Regularization for overlapping particles
		if (dist < 2) {
			regpoly =  3./(4.*PI*dist4) - 3./(64.*PI)*(3. - dist2/2.);
		} else {
			regpoly = 0.;
		}

		// -( (Si*Sj)r + (Sj*r)Si + (Si*r)Sj - 2(Si*r)(Sj*r)r ) term gets the .x field
		h_gradES_table.data[i].x = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);

		// -(Si*r)(Sj*r)r polynomials 
		exppolyp = 9./(1024.*pow(PI,1.5)*xi5*dist4)*(4.*xi4*dist5 - 8.*xi4*dist4 + 8.*xi4*dist3 + 8.*xi2*(1.-2.*xi2)*dist2 + (3.-12.*xi2+32.*xi4)*dist + 2.*(3.+4.*xi2-32.*xi4));
		exppolym = 9./(1024.*pow(PI,1.5)*xi5*dist4)*(4.*xi4*dist5 + 8.*xi4*dist4 + 8.*xi4*dist3 - 8.*xi2*(1.-2.*xi2)*dist2 + (3.-12.*xi2+32.*xi4)*dist - 2.*(3.+4.*xi2-32.*xi4));
		exppoly0 = 9./(512.*pow(PI,1.5)*xi5*dist3)*(-4.*xi4*dist4 + 8.*xi4*dist2 - 3.+36.*xi2);
		erfpolyp = 9./(2048.*PI*xi6*dist4)*(-8.*xi6*dist6 - 4.*xi4*(1.-4.*xi2)*dist4 - 2.*xi2*(1.-8.*xi2)*dist2 + 3.-36.*xi2+256.*xi6);
		erfpolym = erfpolyp;
		erfpoly0 = 9./(1024.*PI*xi6*dist4)*(8.*xi6*dist6 + 4.*xi4*(1.-4.*xi2)*dist4 + 2.*xi2*(1.-8.*xi2)*dist2 - 3.+36.*xi2);

		// Regularization for overlapping particles
		if (dist < 2) {
			regpoly =  -9./(4.*PI*dist4) - 9./(64.*PI)*(1. - dist2/2.);
		} else {
			regpoly = 0.;
		}

		// -(Si*r)(Sj*r)r term gets the .y field
		h_gradES_table.data[i].y = Scalar(exppolyp*expp + exppolym*expm + exppoly0*exp0 + erfpolyp*erfp + erfpolym*erfm + erfpoly0*erf0 + regpoly);
	}

	// Set the .y (or .z and .w) fields of the ith entry to be the value of the .x (or .x and .y) fields of the i+1 entry.  This speeds up linear interpolation later.
	for (int i = 0; i < m_Ntable; i++)
	{
		h_phiS_table.data[i].y = h_phiS_table.data[i+1].x;
		h_ES_table.data[i].z = h_ES_table.data[i+1].x;
		h_ES_table.data[i].w = h_ES_table.data[i+1].y;
		h_gradphiq_table.data[i].y = h_gradphiq_table.data[i+1].x;
		h_gradphiS_table.data[i].z = h_gradphiS_table.data[i+1].x;
		h_gradphiS_table.data[i].w = h_gradphiS_table.data[i+1].y;
		h_gradES_table.data[i].z = h_gradES_table.data[i+1].x;
		h_gradES_table.data[i].w = h_gradES_table.data[i+1].y;
	}

	////// Initializations for needed arrays

	// Group membership list
	GPUArray<int> n_group_membership(m_Ntotal, m_exec_conf);
	m_group_membership.swap(n_group_membership);

	// Particle dipoles
	GPUArray<Scalar3> n_dipole(m_group_size, m_exec_conf);
	m_dipole.swap(n_dipole);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::readwrite);

	// Get access to particle conductivities
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::read);

	// Initialize array for the right side of the linear solve, E0 - M_Eq*q
	GPUArray<Scalar3> n_Eq(m_group_size, m_exec_conf);
	m_Eq.swap(n_Eq);

	// Fill the dipole array
	for( unsigned int ii = 0; ii < m_group_size; ++ii){

		// Compute the beta parameter
		Scalar lambda_p = h_conductivity.data[ii];
		Scalar beta;
		if ( std::isfinite(lambda_p) ){
			beta = (lambda_p - 1.)/(lambda_p + 2.);
		} else {
			beta = 1.;
		}

		// Fill the dipole array with the isolated particle (constant dipole model) dipole
		h_dipole.data[ii] = 4.0*PI*beta*m_field;
	}

}

// Update the applied external field and gradient.  Does not require recomputing tables on the CPU.
void Dielectric2::UpdateField(std::vector<float> &field,
			     std::vector<float> &gradient)
{

	// Set the new field and field gradient
	m_field = make_scalar3(field[0], field[1], field[2]);
	m_gradient = make_scalar3(gradient[0], gradient[1], gradient[2]);
	
	// Get access to the particle conductivity and dipole arrays
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::read);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::readwrite);

	// Update dipoles
	for( unsigned int ii = 0; ii < m_group_size; ++ii){
		
		// Compute the beta parameter
		Scalar lambda_p = h_conductivity.data[ii];
		Scalar beta;
		if ( std::isfinite(lambda_p) ){
			beta = (lambda_p - 1.)/(lambda_p + 2.);
		} else {
			beta = 1.;
		}

		// Fill the dipole array with the new isolated particle (constant dipole model) dipole
		h_dipole.data[ii] = 4.0*PI*beta*m_field;
	}
}

// Update simulation parameters.  Recomputes tables on the CPU.
void Dielectric2::UpdateParameters(std::vector<float> &field,
				   std::vector<float> &gradient,
		      		   std::vector<float> &conductivity,
		      		   std::string fileprefix,
		      		   int period,
				   int dipoleflag,
		      		   unsigned int t0) 
{

	// Extract inputs
	m_field = make_scalar3(field[0], field[1], field[2]);
	m_gradient = make_scalar3(gradient[0], gradient[1], gradient[2]);
	m_fileprefix = fileprefix;
	m_period = period;
	m_dipoleflag = dipoleflag,
	m_t0 = t0;

	// Get access to particle conductivity and dipole arrays
	ArrayHandle<Scalar> h_conductivity(m_conductivity, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::readwrite);

	// Update arrays
	for (unsigned int i = 0; i < m_group_size; ++i ){
		
		// Update particle conductivities
		Scalar lambda_p = conductivity[i];
		h_conductivity.data[i] = lambda_p;

		// Compute the beta parameter
		Scalar beta;
		if ( std::isfinite(lambda_p) ){
			beta = (lambda_p - 1.)/(lambda_p + 2.);
		} else {
			beta = 1.;
		}

		// Fill the dipole array with the new isolated particle (constant dipole model) dipole
		h_dipole.data[i] = 4.0*PI*beta*m_field;
	}
}


// Compute forces on particles
void Dielectric2::computeForces(unsigned int timestep) {

	// access the particle forces (associated with this plugin only; other forces are stored elsewhere)
	ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

	// zero the particle forces
	gpu_ZeroForce(m_Ntotal, d_force.data, 512);

	// update the neighbor list
	m_nlist->compute(timestep);

	// profile this step
	if (m_prof)
		m_prof->push(m_exec_conf, "Dielectric");

	////// Access all of the needed data

	// particle positions
	ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

	// particle conductivities
	ArrayHandle<Scalar> d_conductivity(m_conductivity, access_location::device, access_mode::read);

	// particle charges
	ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);

	// particle dipoles
	ArrayHandle<Scalar3> d_dipole(m_dipole, access_location::device, access_mode::readwrite);

	// active group indices
	ArrayHandle<int> d_group_membership(m_group_membership, access_location::device, access_mode::readwrite);

	// particles in the active group
	ArrayHandle<unsigned int> d_group_members(m_group->getIndexArray(), access_location::device, access_mode::read);

	// simulation box
	BoxDim box = m_pdata->getBox();

	// wave vectors on the grid
	ArrayHandle<Scalar3> d_gridk(m_gridk, access_location::device, access_mode::read);
	
	// scalings on the grid
	ArrayHandle<Scalar> d_scale_phiq(m_scale_phiq, access_location::device, access_mode::read);
	ArrayHandle<Scalar> d_scale_phiS(m_scale_phiS, access_location::device, access_mode::read);
	ArrayHandle<Scalar> d_scale_ES(m_scale_ES, access_location::device, access_mode::read);
	
	// wave space grids
	ArrayHandle<CUFFTCOMPLEX> d_phiq_grid(m_phiq_grid, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_phiS_grid(m_phiS_grid, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_Eq_gridX(m_Eq_gridX, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_Eq_gridY(m_Eq_gridY, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_Eq_gridZ(m_Eq_gridZ, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_ES_gridX(m_ES_gridX, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_ES_gridY(m_ES_gridY, access_location::device, access_mode::readwrite);
	ArrayHandle<CUFFTCOMPLEX> d_ES_gridZ(m_ES_gridZ, access_location::device, access_mode::readwrite);

	// real space tables
	ArrayHandle<Scalar2> d_phiS_table(m_phiS_table, access_location::device, access_mode::read);
	ArrayHandle<Scalar4> d_ES_table(m_ES_table, access_location::device, access_mode::read);
	ArrayHandle<Scalar2> d_gradphiq_table(m_gradphiq_table, access_location::device, access_mode::read);
	ArrayHandle<Scalar4> d_gradphiS_table(m_gradphiS_table, access_location::device, access_mode::read);
	ArrayHandle<Scalar4> d_gradES_table(m_gradES_table, access_location::device, access_mode::read);

	// right hand side array for the linear solve M_ES*S = E0 - M_Eq*q
	ArrayHandle<Scalar3> d_Eq(m_Eq, access_location::device, access_mode::readwrite);

	// neighbor list
	ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(), access_location::device, access_mode::read);

	// index in neighbor list where each particle's neighbors begin
	ArrayHandle<unsigned int> d_head_list(m_nlist->getHeadList(), access_location::device, access_mode::read);

	// number of neighbors of each particle
	ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(), access_location::device, access_mode::read);

	// set the block size of normal GPU kernels
	int block_size = 512;

	// perform the calculation on the GPU
	if (m_dipoleflag != 2) {
		gpu_ComputeForce(d_pos.data, d_group_membership.data, m_Ntotal, d_group_members.data, m_group_size, box, block_size, d_force.data, d_charge.data, 
				 d_conductivity.data, d_dipole.data, m_field, m_gradient, d_Eq.data, m_xi, m_eta, m_rc, m_drtable, m_Ntable, d_phiS_table.data, 
				 d_ES_table.data, d_gradphiq_table.data, d_gradphiS_table.data, d_gradES_table.data, d_gridk.data, d_scale_phiq.data, d_scale_phiS.data, 
				 d_scale_ES.data ,d_phiq_grid.data, d_phiS_grid.data, d_Eq_gridX.data, d_Eq_gridY.data, d_Eq_gridZ.data, d_ES_gridX.data,
				 d_ES_gridY.data, d_ES_gridZ.data, m_plan, m_Nx, m_Ny, m_Nz, d_n_neigh.data, d_nlist.data, d_head_list.data, m_P, m_gridh, m_errortol, 
				 m_dipoleflag);
	} else {
		gpu_ComputeForce_Charge(d_pos.data, d_group_membership.data, m_Ntotal, d_group_members.data, m_group_size, box, block_size, d_force.data, d_charge.data, 
				  	m_field, m_xi, m_eta, m_rc, m_drtable, m_Ntable, d_gradphiq_table.data, d_scale_phiq.data, d_phiq_grid.data, m_plan, m_Nx, m_Ny,
					m_Nz, d_n_neigh.data, d_nlist.data, d_head_list.data, m_P, m_gridh, m_errortol);
	}

	if (m_exec_conf->isCUDAErrorCheckingEnabled())
		CHECK_CUDA_ERROR();

	// done profiling
	if (m_prof)
		m_prof->pop(m_exec_conf);

	// If the period is set, create a file every period timesteps.
	if ( ( m_period > 0 ) && ( (int(timestep) - m_t0) % m_period == 0 ) ) {
		OutputData(int(timestep));
	}
}

// Write quantities to file
void Dielectric2::OutputData(unsigned int timestep) {

	// Format the timestep to a string
	std::ostringstream timestep_str;
	timestep_str << std::setw(10) << std::setfill('0') << timestep;

	// Construct the filename
	std::string filename = m_fileprefix + "." + timestep_str.str() + ".txt";

	// Access needed data
	ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
	ArrayHandle<Scalar3> h_dipole(m_dipole, access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
	ArrayHandle<int> h_group_membership(m_group_membership, access_location::host, access_mode::read);

	// Open the file
	std::ofstream file;
	file.open(filename.c_str(), std::ios_base::out);

	// Check that the file opened correctly
        if (!file.good()) {
                throw std::runtime_error("Error in Dielectric2: unable to open output file.");
        }

	////// Write the particle positions to file in global tag order

	// Header
	file << "Position" << std::endl;

	// Loop through particle tags
	for (int i = 0; i < m_Ntotal; i++) {

		// Get the particle's global index
		unsigned int idx = h_rtag.data[i];

		// Get the particle's position
		Scalar4 postype = h_pos.data[idx];

		// Write the position to file
		file << std::setprecision(10) << postype.x << "  " << postype.y << "  " << postype.z << "  " << std::endl;
	}

	////// Write the particle dipoles to file in global tag order
	file << "Dipole" << std::endl;
	for (int i = 0; i < m_Ntotal; i++) {

		// Get the particle's global index
		unsigned int idx = h_rtag.data[i];

		// Get the particle's active group-specific index
		int group_idx = h_group_membership.data[idx];

		// Get the particle's dipole if it is in the active group.  Else, set the dipole to 0.
		Scalar3 dipole;
		if (group_idx != -1) {
			dipole = h_dipole.data[group_idx];
		} else {
			dipole = make_scalar3(0.0, 0.0, 0.0);
		}

		// Write the dipole to file
		file << std::setprecision(10) << dipole.x << "  " << dipole.y << "  " << dipole.z << "  " << std::endl;
	}

	////// Write the particle electric/magnetic forces to file in global tag order
	file << "Force" << std::endl;
	for (int i = 0; i < m_Ntotal; i++) {

		// Get the particle's global index
		unsigned int idx = h_rtag.data[i];

		// Get the particle's electric/magnetic force
		Scalar4 force = h_force.data[idx];

		// Write the dipole to file
		file << std::setprecision(10) << force.x << "  " << force.y << "  " << force.z << "  " << std::endl;
	}


	// Close output file
	file.close();
}

void export_Dielectric2(pybind11::module& m)
{
    pybind11::class_<Dielectric2, std::shared_ptr<Dielectric2>> (m, "Dielectric2", pybind11::base<ForceCompute>())
		.def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<NeighborList>, std::vector<float>&, std::vector<float>&, std::vector<float>&, Scalar, Scalar, std::string, int, int, unsigned int >())
		.def("SetParams", &Dielectric2::SetParams)
		.def("UpdateField", &Dielectric2::UpdateField)
		.def("UpdateParameters", &Dielectric2::UpdateParameters)
		.def("computeForces", &Dielectric2::computeForces)
		.def("OutputData", &Dielectric2::OutputData)
        ;
}

#ifdef WIN32
#pragma warning( pop )
#endif

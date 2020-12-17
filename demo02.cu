#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "definitions.h"
#include "particle.h"
#include "beam_elements.h"
#include "beamfields.h"
#include "fodo_lattice.h"

__global__ void Track_particles_until_turn(
    demotrack::Particle* particle_set,
    demotrack::int64_type const num_particles,
    double const* __restrict__ lattice_buffer,
    demotrack::uint64_type const max_lattice_buffer_index,
    demotrack::int64_type const until_turn )
{
    namespace dt = demotrack;

    dt::int64_type const STRIDE = blockDim.x * gridDim.x;
    dt::int64_type idx = threadIdx.x + blockIdx.x * blockDim.x;

    for( ; idx < num_particles ; idx += STRIDE )
    {
        dt::Particle p = particle_set[ idx ];
        dt::uint64_type const start_at_element = p.at_element;

        while( ( p.state == 1 ) && ( p.at_turn < until_turn ) )
        {
            dt::uint64_type slot_idx = 0;

            while( ( p.state == 1 ) && ( slot_idx < max_lattice_buffer_index ) )
            {
                /* all elements are stored with their type_id as the first
                 * data member -> retrieve this number and dispatch
                 * the track method accordingly */

                dt::beam_element_type const type_id = ( dt::beam_element_type )(
                    int )lattice_buffer[ slot_idx ];

                switch( type_id )
                {
                    case dt::BEAM_ELEMENT_DRIFT: // cf. beam_elements.h
                    {
                        const dt::Drift *const __restrict__ elem =
                            ( dt::Drift const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        dt::Drift::GLOBAL_APERTURE_CHECK( p );
                        slot_idx = next_slot_idx;
                        break;
                    }

                    case dt::BEAM_ELEMENT_DRIFT_EXACT: // cf. beam_elements.h
                    {
                        const dt::DriftExact *const __restrict__ elem =
                            ( dt::DriftExact const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        // Use GLOBAL_APERTURE_CHECK from Drift -> it's the same
                        dt::Drift::GLOBAL_APERTURE_CHECK( p );
                        slot_idx = next_slot_idx;
                        break;
                    }

                    case dt::BEAM_ELEMENT_MULTIPOLE: // cf. beam_elements.h
                    {
                        const dt::Multipole *const __restrict__ elem =
                            ( dt::Multipole const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        slot_idx = next_slot_idx;
                        break;
                    }

                    case dt::BEAM_ELEMENT_XY_SHIFT: // cf. beam_elements.h
                    {
                        const dt::XYShift *const __restrict__ elem =
                            ( dt::XYShift const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        slot_idx = next_slot_idx;
                        break;
                    }

                    case dt::BEAM_ELEMENT_SROTATION: // cf. beam_elements.h
                    {
                        const dt::SRotation *const __restrict__ elem =
                            ( dt::SRotation const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        slot_idx = next_slot_idx;
                        break;
                    }

                    case dt::BEAM_ELEMENT_CAVITY: // cf. beam_elements.h
                    {
                        const dt::Cavity *const __restrict__ elem =
                            ( dt::Cavity const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        slot_idx = next_slot_idx;
                        break;
                    }

                    #if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && \
                        DEMOTRACK_ENABLE_BEAMFIELDS == 1

                    case dt::BEAM_ELEMENT_SC_COASTING: // cf. beamfields.h
                    {
                        const dt::SpaceChargeCoasting *const __restrict__ elem =
                            ( dt::SpaceChargeCoasting const* )&lattice_buffer[ slot_idx ];

                        dt::uint64_type const next_slot_idx =
                            elem->track( p, slot_idx );

                        slot_idx = next_slot_idx;
                        break;
                    }

                    #endif /* beamfields enabled */

                    default:
                    {
                        /* unknown beam element -> loose particle and quit */
                        p.state = 0;
                        slot_idx = max_lattice_buffer_index;
                    }
                };

            }

            if( p.state == 1 )
            {
                p.at_element = start_at_element;
                ++p.at_turn;
            }
        }

        particle_set[ idx ] = p;
    }
}

int main( int argc, char* argv[] )
{
    namespace dt = demotrack;

    /* ********************************************************************* */
    /* Prepare particle set to track */

    dt::uint64_type NUM_PARTICLES = 50 * 1024;
    dt::int64_type  TRACK_UNTIL_TURN = 1000;

    if( argc >= 2 )
    {
        NUM_PARTICLES = std::stoi( argv[ 1 ] );

        if( argc >= 3 )
        {
            TRACK_UNTIL_TURN = std::stoi( argv[ 2 ] );
        }
    }
    else
    {
        std::cout << "Usage : " << argv[ 0 ]
                  << " [NUM_PARTICLES] [TRACK_UNTIL_TURN]\r\n"
                  << std::endl;
    }

    double const P0_C    = 470e9;  /* Kinetic energy, [eV]  */
    double const MASS0   = 938.272081e6; /* Proton rest mass, [eV] */
    double const CHARGE0 = 1.0; /* Reference particle charge; here == proton */
    double const DELTA   = 0.0; /* momentum deviation from reference particle */

    std::vector< dt::Particle > particles_host( NUM_PARTICLES );

    dt::uint64_type particle_id = 0u;
    for( auto& p : particles_host )
    {
        p.init( MASS0, CHARGE0, P0_C, DELTA );
        p.id = particle_id++;
    }

    /* ********************************************************************* */
    /* Prepare lattice / machine description: */

    double fodo_lattice[ 200 ];

    /* see fodo_lattice.h for the implementation of create_fodo_lattice */
    dt::uint64_type const LATTICE_SIZE =
        dt::create_fodo_lattice( &fodo_lattice[ 0 ], 200u );

    /* ********************************************************************** */
    /* Allocate buffers on the device */

    dt::Particle* particles_dev = nullptr;
    double* lattice_dev = nullptr;

    auto status = ::cudaMalloc( ( void** )&particles_dev,
        sizeof( dt::Particle ) * NUM_PARTICLES );
    assert( status == CUDA_SUCCESS );

    status = ::cudaMalloc( ( void** )&lattice_dev,
        LATTICE_SIZE * sizeof( double ) );
    assert( status == CUDA_SUCCESS );

    /* Copy particle and lattice data to device */

    status = ::cudaMemcpy( lattice_dev, &fodo_lattice[ 0 ],
        LATTICE_SIZE * sizeof( double ), ::cudaMemcpyHostToDevice );
    assert( status == CUDA_SUCCESS );

    status = ::cudaMemcpy( particles_dev, particles_host.data(),
        particles_host.size() * sizeof( dt::Particle ),
            ::cudaMemcpyHostToDevice );

    assert( status == CUDA_SUCCESS );

    /* ******************************************************************** */
    /* Estimate block size */

    int BLOCK_SIZE = 0;
    int MIN_GRID_SIZE = 0;

    status = ::cudaOccupancyMaxPotentialBlockSize(
        &MIN_GRID_SIZE, /* -> minimum grid size needed for max occupancy */
        &BLOCK_SIZE, /* -> estimated optimal block size */
        Track_particles_until_turn, /* the kernel */
        0u, /* -> dynamic shared memory per block required [bytes] */
        0u /* -> max block size limit for the kernel; 0 == no limit */ );

    assert( status == CUDA_SUCCESS );

    assert( BLOCK_SIZE > 0 );
    int const GRID_SIZE = ( NUM_PARTICLES + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

    /* ******************************************************************** */
    /* Run kernel: */

    ::cudaDeviceProp props;
    int device = 0;
    status = ::cudaGetDevice( &device );
    assert( status == CUDA_SUCCESS );

    status = ::cudaGetDeviceProperties( &props, device );
    assert( status == CUDA_SUCCESS );

    char pci_bus_id_str[] =
    {
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
    };

    status = ::cudaDeviceGetPCIBusId( pci_bus_id_str, 32, device );
    assert( status == CUDA_SUCCESS );

    std::cout << "number of particles   : " << NUM_PARTICLES << "\r\n"
              << "number of turns       : " << TRACK_UNTIL_TURN << "\r\n";

    #if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && DEMOTRACK_ENABLE_BEAMFIELDS == 1
    std::cout << "space-charge enabled  : true\r\n";
    #else
    std::cout << "space-charge enabled  : false\r\n";
    #endif /* SC emabled */

    std::cout << "DEVICE                : " << pci_bus_id_str
              << " ( " << props.name << " )\r\n"
              << "NUM_OF_BLOCKS         : " << GRID_SIZE << "\r\n"
              << "THREADS_PER_BLOCK     : " << BLOCK_SIZE << "\r\n";

    /* Prepare cuda events to estimate the elapsed wall time */

    ::cudaEvent_t start;
    status = ::cudaEventCreate( &start );
    assert( status == CUDA_SUCCESS );

    ::cudaEvent_t stop;
    status = ::cudaEventCreate( &stop );
    assert( status == CUDA_SUCCESS );

    status = ::cudaEventRecord( start );
    assert( status == CUDA_SUCCESS );

    /* Run kernel */

    Track_particles_until_turn<<< GRID_SIZE, BLOCK_SIZE >>>(
        particles_dev, NUM_PARTICLES, lattice_dev, LATTICE_SIZE,
            TRACK_UNTIL_TURN );
    status = ::cudaDeviceSynchronize();

    /* Estimate wall time */

    status = ::cudaEventRecord( stop );
    assert( status == CUDA_SUCCESS );

    status = ::cudaEventSynchronize( stop );
    assert( status == CUDA_SUCCESS );

    float wtime = 0.0;
    status = ::cudaEventElapsedTime( &wtime, start, stop );
    assert( status == CUDA_SUCCESS );

    std::cout << "-------------------------------------------------------\r\n"
              << "Elapsed time          : " << wtime << " msec total \r\n"
              << "                      : " << wtime / (
                std::max( NUM_PARTICLES * TRACK_UNTIL_TURN,
                          dt::uint64_type{ 1 } ) )
              << " msec / particle / turn\r\n";

    /* Fetch data */

    status = ::cudaMemcpy( particles_host.data(), particles_dev,
                           particles_host.size() * sizeof( dt::Particle ),
                           ::cudaMemcpyDeviceToHost );
    assert( status == CUDA_SUCCESS );

    /* ********************************************************************* */
    /* Verify tracking results */

    dt::uint64_type num_active_particles = 0u;
    dt::uint64_type num_lost_particles = 0u;

    for( auto& p : particles_host )
    {
        if( ( p.state == 1 ) && ( p.at_turn == TRACK_UNTIL_TURN ) )
        {
            ++num_active_particles;
        }
        else if( ( p.state == 0 ) && ( p.at_turn < TRACK_UNTIL_TURN ) )
        {
            ++num_lost_particles;
        }
        else
        {
            std::cerr << "illegal particle id = " << p.id
                      << ", at_turn = " << p.at_turn
                      << ", at_element = " << p.at_element
                      << ", state = " << p.state << std::endl;
        }
    }

    std::cout << "-------------------------------------------------------\r\n"
              << "num lost particles    : " << num_lost_particles << "\r\n"
              << "num active particles  : " << num_active_particles << "\r\n"
              << std::endl;

    /* ********************************************************************* */
    /* Cleaning up, Freeing resources */

    ::cudaFree( lattice_dev );
    lattice_dev = nullptr;

    ::cudaFree( particles_dev );
    particles_dev = nullptr;

    ::cudaEventDestroy( start );
    ::cudaEventDestroy( stop );

    return 0;
}


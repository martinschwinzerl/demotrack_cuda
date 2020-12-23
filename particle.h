#ifndef DEMOTRACK_CUDA_PARTICLE_H__
#define DEMOTRACK_CUDA_PARTICLE_H__

#include "definitions.h"

#include <cmath>
#include <vector>
#include <string>

namespace demotrack
{
    struct Particle
    {
        DEMOTRACK_FN void init( double const mass0, double const charge0,
            double const p0c, double const delta ) noexcept {
            using std::sqrt;

            double const energy0 = sqrt( p0c * p0c + mass0 * mass0 );
            double const beta0 = p0c / energy0;
            double const beta0_squ = beta0 * beta0;
            double const gamma0 = 1. / sqrt( 1.0 - beta0_squ );
            double const ptau = sqrt( delta * delta + 2. * delta +
                1. / beta0_squ ) - 1. / beta0;
            double const psigma = ptau / beta0;

            this->x            = 0.;
            this->px           = 0.;
            this->y            = 0.;
            this->py           = 0.;
            this->zeta         = 0.;
            this->delta        = delta;

            this->rpp          = 1.;
            this->rvv          = 1.;
            this->psigma       = psigma;
            this->chi          = 1.;

            this->charge_ratio = 1.;

            this->charge0      = charge0;
            this->mass0        = mass0;
            this->beta0        = beta0;
            this->gamma0       = gamma0;
            this->p0c          = p0c;

            this->state        = 1;
            this->at_element   = 0;
            this->at_turn      = 0;
            this->id           = 0;
        }

        DEMOTRACK_FN double energy0() const noexcept {
            using std::sqrt;
            return sqrt( this->p0c * this->p0c + this->mass0 * this->mass0 );
        }

        DEMOTRACK_FN void add_to_energy( double const dE ) noexcept {
            using std::sqrt;

            double const delta_beta0 = this->delta * this->beta0;
            double const ptau_beta0 = dE / this->energy0() + sqrt( delta_beta0 *
                delta_beta0 + 2. * delta_beta0 * this->beta0 + 1. ) - 1.;

            double const ptau = ptau_beta0 / this->beta0;
            double const psigma = ptau / this->beta0;
            double const delta = sqrt( ptau * ptau + 2. * psigma + 1. ) - 1.;

            double const one_plus_delta = delta + 1.;
            double const rvv = one_plus_delta / ( 1. + ptau_beta0 );

            this->delta = delta;
            this->psigma = psigma;
            this->zeta *= rvv / this->rvv;
            this->rvv = rvv;
            this->rpp = 1. / one_plus_delta;
        }

        double x;
        double px;
        double y;
        double py;
        double zeta;
        double delta;

        double rpp;
        double rvv;
        double psigma;
        double chi;
        double charge_ratio;

        double charge0;
        double mass0;
        double beta0;
        double gamma0;
        double p0c;

        uint64_type state;
        uint64_type at_element;
        int64_type  at_turn;
        uint64_type id;
    };

    DEMOTRACK_HOST_FN void Particles_load(
        std::vector< demotrack::Particle >& __restrict__ particles_data,
        uint64_type const num_of_particles,
        std::string& __restrict__ path_to_particle_data,
        double const P0C = double{ 470e9 },
        double const MASS0 = double{ 938.272081e6 },
        double const CHARGE0 = double{ 1. }, double const DELTA = double{ 0. } )
    {
        namespace dt = demotrack;
        particles_data.clear();

        if( num_of_particles == 0u ) return;

        if( !path_to_particle_data.empty() )
        {
            ::FILE* fp = std::fopen( path_to_particle_data.c_str(), "rb" );

            if( fp != nullptr )
            {
                double temp;
                dt::uint64_type ret = std::fread(
                    &temp, sizeof( double ), 1u, fp );

                dt::uint64_type num_avail_particles = ( temp > 0.0 )
                    ? static_cast< dt::uint64_type >( temp )
                    : dt::uint64_type{ 0 };

                if( ( ret == 1u ) && ( num_avail_particles > 0u ) )
                {
                    std::vector< dt::Particle > temp_particle_buffer(
                        num_avail_particles, dt::Particle{} );

                    for( dt::uint64_type ii = 0u ;
                         ii < num_avail_particles ; ++ii )
                    {
                        ret = std::fread( &( temp_particle_buffer.back() ),
                            sizeof( dt::Particle ), 1u, fp );

                        if( ret != 1u )
                        {
                            auto it = temp_particle_buffer.begin();
                            auto end = it;
                            std::advance( it, ii );

                            particles_data.assign( it, end );
                            temp_particle_buffer.clear();
                            temp_particle_buffer.assign(
                                particles_data.cbegin(),
                                particles_data.cend() );
                            particles_data.clear();
                            num_avail_particles = ii;
                            break;
                        }
                    }

                    if( num_avail_particles > 0u )
                    {
                        if( num_avail_particles == num_of_particles )
                        {
                            temp_particle_buffer.swap( particles_data );
                        }
                        else
                        {
                            particles_data.resize(
                                num_of_particles, dt::Particle{} );

                            dt::uint64_type ii = 0u;
                            for( ; ii < num_of_particles ; ++ii )
                            {
                                particles_data[ ii ] = temp_particle_buffer[
                                    ii % num_avail_particles ];
                                particles_data[ ii ].id = ii;
                            }
                        }
                    }
                }
            }
        }

        if( particles_data.size() < num_of_particles )
        {
            if( ( !path_to_particle_data.empty() ) &&
                ( particles_data.empty() ) )
            {
                path_to_particle_data.clear();
            }

            particles_data.resize( num_of_particles, dt::Particle{} );
            dt::uint64_type particle_id = 0u;

            for( auto& p : particles_data )
            {
                p.init( MASS0, CHARGE0, P0C, DELTA );
                p.id = particle_id++;
            }
        }
    }

    DEMOTRACK_HOST_FN void Particles_load(
        std::vector< demotrack::Particle >& __restrict__ particles_data,
        uint64_type const num_of_particles,
        double const P0C, double const MASS0,
        double const CHARGE0 = double{ 1.0 },
        double const DELTA = double{ 0. } ) {
        std::string dummy_str = std::string{};
        demotrack::Particles_load( particles_data, num_of_particles, dummy_str,
            P0C, MASS0, CHARGE0, DELTA );
    }
}

#endif /* DEMOTRACK_CUDA_PARTICLE_H__ */

#include "definitions.h"
#include "particle.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main( int const argc, char* argv[] )
{
    namespace dt = demotrack;
    std::string path_to_particles_data = std::string{};
    dt::uint64_type num_particles = dt::uint64_type{ 0 };

    double P0C = double{ 470e9 };
    double MASS0 = double{ 938.272081e6 };
    double CHARGE0 = double{ 1.0 };

    double MIN_X = double{ 0.0 };
    double MAX_X = double{ 0.0 };

    double MIN_Y = double{ 0.0 };
    double MAX_Y = double{ 0.0 };

    double MIN_PX = double{ 0.0 };
    double MAX_PX = double{ 0.0 };

    double MIN_PY = double{ 0.0 };
    double MAX_PY = double{ 0.0 };

    double MIN_ZETA = double{ 0.0 };
    double MAX_ZETA = double{ 0.0 };

    double MIN_DELTA = double{ 0.0 };
    double MAX_DELTA = double{ 0.0 };

    if( argc >= 5 )
    {
        path_to_particles_data = std::string{ argv[ 1 ] };
        num_particles = std::stoul( argv[ 2 ] );
        P0C = std::max( std::stod( argv[ 3 ] ), double{ 0. } );
        MASS0 = std::max( std::stod( argv[ 4 ] ), double{ 0. } );

        if( argc >= 6 )
        {
            CHARGE0 = std::stod( argv[ 5 ] );

            if( argc >= 8 )
            {
                MIN_X = std::stod( argv[ 6 ] );
                MAX_X = std::stod( argv[ 7 ] );
            }

            if( argc >= 10 )
            {
                MIN_X = std::stod( argv[ 8 ] );
                MAX_X = std::stod( argv[ 9 ] );
            }

            if( argc >= 12 )
            {
                MIN_PX = std::stod( argv[ 10 ] );
                MAX_PX = std::stod( argv[ 11 ] );
            }

            if( argc >= 14 )
            {
                MIN_PY = std::stod( argv[ 12 ] );
                MAX_PY = std::stod( argv[ 13 ] );
            }

            if( argc >= 16 )
            {
                MIN_ZETA = std::stod( argv[ 14 ] );
                MAX_ZETA = std::stod( argv[ 15 ] );
            }

            if( argc >= 18 )
            {
                MIN_DELTA = std::stod( argv[ 16 ] );
                MAX_DELTA = std::stod( argv[ 17 ] );
            }
        }
    }
    else
    {
        std::cout << "Usage: " << argv[ 0 ] << " PATH_TO_PARTICLE_DATA_DUMP "
                  << "NUM_PARTICLES P0C MASS0 [CHARGE0]\r\n"
                  << "            [MIN_X]    [MAX_X]    [MIN_Y]     [MAX_Y]\r\n"
                  << "            [MIN_PX]   [MAX_PX]   [MIN_PY]    [MAX_PY]\r\n"
                  << "            [MIN_ZETA] [MAX_ZETA] [MIN_DELTA] [MAX_DELTA]\r\n"
                  << std::endl;

        return 0;
    }

    std::unique_ptr<::FILE, decltype(&std::fclose)> fp(
        std::fopen( path_to_particles_data.c_str(), "wb" ), &std::fclose );

    if( ( fp.get() != nullptr ) && ( num_particles > dt::uint64_type{ 0 } ) )
    {
        double const denom = double{ 1. } / static_cast< double >(
            num_particles - 1 );

        double const dX     = ( MAX_X     - MIN_X     ) * denom;
        double const dY     = ( MAX_Y     - MIN_Y     ) * denom;
        double const dPx    = ( MAX_PX    - MIN_PX    ) * denom;
        double const dPy    = ( MAX_PY    - MIN_PY    ) * denom;
        double const dZeta  = ( MAX_ZETA  - MIN_ZETA  ) * denom;
        double const dDelta = ( MAX_DELTA - MIN_DELTA ) * denom;

        double temp = static_cast< double >( num_particles );

        auto ret = std::fwrite( &temp, sizeof( double ), 1u, fp.get() );

        bool success = ( ret == 1u );
        dt::uint64_type ii = 0u;

        while( ( success ) && ( ii < num_particles ) )
        {
            double const delta_ii = MIN_DELTA + dDelta * ii;

            dt::Particle p;
            p.init( MASS0, CHARGE0, P0C, delta_ii );
            p.x  = MIN_X  + dX * ii;
            p.y  = MIN_Y  + dY * ii;
            p.px = MIN_PX + dPx * ii;
            p.py = MIN_PY + dPy * ii;
            p.zeta = MIN_ZETA + dZeta * ii;
            p.id = ii++;
            p.at_element = 0;
            p.at_turn = 0;
            p.state = 1;

            ret = std::fwrite( &p, sizeof( dt::Particle ), 1u, fp.get() );

            if( ret != 1u )
            {
                success = false;
                break;
            }
        }

        if( !success )
        {
            fp.reset( nullptr );
            std::remove( path_to_particles_data.c_str() );
        }

        std::fflush( fp.get() );
    }

    return 0;
}


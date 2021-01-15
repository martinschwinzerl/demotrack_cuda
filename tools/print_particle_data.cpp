#include "definitions.h"
#include "particle.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

int main( int const argc, char* argv[] )
{
    namespace dt = demotrack;
    std::string path_to_datafile = std::string{};
    bool print_active = true;
    bool print_lost = true;

    if( argc >= 2 )
    {
        path_to_datafile = std::string{ argv[ 1 ] };

        if( argc >= 3 )
        {
            std::string const filter = std::string{ argv[ 2 ] };

            if( filter.compare( "lost" ) == 0 )
            {
                print_active = false;
            }
            else if( filter.compare( "active" ) == 0 )
            {
                print_lost = false;
            }
        }
    }
    else
    {
        std::cout << argv[ 0 ] << " PATH_TO_PARTICLES_DATAFILE [FILTER]\r\n"
                  << std::endl;

        return 0;
    }

    std::unique_ptr<::FILE, decltype(&std::fclose)> fp(
        std::fopen( path_to_datafile.c_str(), "rb" ), &std::fclose );

    if( fp.get() == nullptr )
    {
        std::cout << "unable to open datafile: " << path_to_datafile << "\r\n"
                  << std::endl;

        return 0;
    }

    double temp_npart = double{ -1.0 };
    auto ret = std::fread( &temp_npart, sizeof( double ), 1u, fp.get() );
    assert( ret == 1u );

    if( ( ret != 1u ) || ( temp_npart <= double{ 0. } ) )
    {
        std::cout << "illegal number of particles in data file\r\n"
                  << std::endl;
        return 0;
    }

    std::size_t const npart = static_cast< std::size_t >( temp_npart );
    dt::Particle p;

    std::cout << "#     idx "
              << "   at_element "
              << "      at_turn "
              << "        state "
              << "                  x "
              << "                 px "
              << "                  y "
              << "                 py "
              << "               zeta "
              << "              delta "
              << "       psigma "
              << "          rpp "
              << "          rvv "
              << "          chi "
              << " charge_ratio "
              << "      charge0 "
              << "        mass0 "
              << "        beta0 "
              << "       gamma0 "
              << "          p0c "
              << "           id \r\n";

    for( std::size_t ii = 0u ; ii < npart ; ++ii )
    {
        ret = std::fread( &p, sizeof( dt::Particle ), 1u, fp.get() );
        if( ret != 1u ) continue;

        if( ( ( p.state == 1 ) && ( print_active ) ) ||
            ( ( p.state == 0 ) && ( print_lost   ) ) )
        {
            std::cout << std::setw( 10 ) << ii
                      << std::setw( 14 ) << p.at_element
                      << std::setw( 14 ) << p.at_turn
                      << std::setw( 14 ) << p.state
                      << std::setw( 20 ) << p.x
                      << std::setw( 20 ) << p.px
                      << std::setw( 20 ) << p.y
                      << std::setw( 20 ) << p.py
                      << std::setw( 20 ) << p.zeta
                      << std::setw( 20 ) << p.delta
                      << std::setw( 14 ) << p.psigma
                      << std::setw( 14 ) << p.rpp
                      << std::setw( 14 ) << p.rvv
                      << std::setw( 14 ) << p.chi
                      << std::setw( 14 ) << p.charge_ratio
                      << std::setw( 14 ) << p.charge0
                      << std::setw( 14 ) << p.mass0
                      << std::setw( 14 ) << p.beta0
                      << std::setw( 14 ) << p.gamma0
                      << std::setw( 14 ) << p.p0c
                      << std::setw( 14 ) << p.id << "\r\n";
        }
    }

    return 0;
}

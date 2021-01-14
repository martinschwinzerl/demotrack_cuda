#include "definitions.h"
#include "particle.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <cmath>
#include <string>
#include <vector>

namespace demotrack
{
    bool are_close( double const lhs, double const rhs,
        double const abs_tol, double const rel_tol )
    {
        return std::fabs( lhs -rhs ) < ( abs_tol + rel_tol * std::fabs( rhs ) );
    }

    bool are_equal( double const lhs, double const rhs ) {
        return demotrack::are_close( lhs, rhs,
            std::numeric_limits< double >::epsilon(), double{ 0. } ); }
}

int main( int const argc, char* argv[] )
{
    namespace dt = demotrack;
    std::string path_to_lhs = std::string{};
    std::string path_to_rhs = std::string{};
    static constexpr double EPS = std::numeric_limits< double >::epsilon();
    double abs_tol = EPS;
    double rel_tol = 0.0;
    bool stop_after_first_diff = false;

    if( argc >= 3 )
    {
        path_to_lhs = std::string{ argv[ 1 ] };
        path_to_rhs = std::string{ argv[ 2 ] };

        if( argc >= 4 )
        {
            abs_tol = std::stod( std::string{ argv[ 3 ] } );

            if( argc >= 5 )
            {
                rel_tol = std::stod( std::string{ argv[ 4 ] } );

                if( argc >= 6 )
                {
                    stop_after_first_diff = (
                        std::stoi( std::string{ argv[ 5 ] } ) == 1 );
                }
            }
        }
    }

    if( argc < 3 )
    {
        std::cout << argv[ 0 ] << " PATH_TO_LHS_PARTICLES "
                  << " PATH_TO_RHS_PARTICLES [ABS_TOL] [REL_TOL]"
                  << " [STOP_AFTER_FIRST_DIFF]\r\n" << std::endl;
        return 1;
    }

    if( ( path_to_lhs.empty() ) || ( path_to_rhs.empty() ) ||
        ( path_to_lhs.compare( path_to_rhs ) == 0 ) )
    {
        std::cout << "ERROR   :: two particle data files requiured\r\n";
        return 1;
    }

    std::unique_ptr<::FILE, decltype(&std::fclose)> fp_lhs(
        std::fopen( path_to_lhs.c_str(), "rb" ), &std::fclose );

    std::unique_ptr<::FILE, decltype(&std::fclose)> fp_rhs(
        std::fopen( path_to_rhs.c_str(), "rb" ), &std::fclose );

    if( ( fp_lhs.get() == nullptr ) || ( fp_rhs.get() == nullptr ) )
    {
        std::cout << "ERROR   :: two particle data files requiured\r\n";
        return 1;
    }

    double lhs_npart = double{ -1.0 };
    double rhs_npart = double{ -1.0 };

    auto ret_lhs = std::fread( &lhs_npart, sizeof( double ), 1u, fp_lhs.get() );
    auto ret_rhs = std::fread( &rhs_npart, sizeof( double ), 1u, fp_rhs.get() );

    if( ( ret_lhs != 1u ) || ( lhs_npart < double{ 0.0 } ) )
    {
        std::cout << "ERROR   :: unable to read number of particles from lhs file"
                  << std::endl;
        return 1;
    }

    if( ( ret_rhs != 1u ) || ( rhs_npart < double{ 0.0 } ) )
    {
        std::cout << "ERROR   :: unable to read number of particles from rhs file"
                  << std::endl;
        return 1;
    }

    if( !dt::are_equal( lhs_npart, rhs_npart ) )
    {
        std::cout << "ERROR   :: difference in number of particles for "
                  << "lhs and rhs files\r\n"
                  << "ERROR   :: lhs: npart = " << lhs_npart << "\r\n"
                  << "ERROR   :: rhs: npart = " << rhs_npart << "\r\n";

        return 1;
    }

    std::size_t const NUM_PARTICLES = static_cast< std::size_t >( lhs_npart );
    bool success = true;

    for( std::size_t ii = 0u ; ii < NUM_PARTICLES ; ++ii )
    {
        dt::Particle lhs_p;
        dt::Particle rhs_p;

        /* ----------------------------------------------------------------- */
        ret_lhs = std::fread( &lhs_p, sizeof( dt::Particle ), 1u, fp_lhs.get() );

        if( ret_lhs != 1u )
        {
            std::cout << "ERROR   :: unable to read particle #" << ii
                      << "from lhs data file\r\n";
            success = false;
            break;
        }

        /* ----------------------------------------------------------------- */
        ret_rhs = std::fread( &rhs_p, sizeof( dt::Particle ), 1u, fp_rhs.get() );

        if( ret_rhs != 1u )
        {
            std::cout << "ERROR   :: unable to read particle #" << ii
                      << "from rhs data file\r\n";
            success = false;
            break;
        }

        /* ----------------------------------------------------------------- */

        if( ( !dt::are_close( lhs_p.x,      rhs_p.x,      abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.px,     rhs_p.px,     abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.y,      rhs_p.y,      abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.py,     rhs_p.py,     abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.zeta,   rhs_p.zeta,   abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.delta,  rhs_p.delta,  abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.rpp,    rhs_p.rpp,    abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.rvv,    rhs_p.rvv,    abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.psigma, rhs_p.psigma, abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.chi,    rhs_p.chi,    abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.charge_ratio, rhs_p.charge_ratio,
                              abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.charge0, rhs_p.charge0, abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.mass0,   rhs_p.mass0,   abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.beta0,   rhs_p.beta0,   abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.gamma0,  rhs_p.gamma0,  abs_tol, rel_tol ) ) ||
            ( !dt::are_close( lhs_p.p0c,     rhs_p.p0c,     abs_tol, rel_tol ) ) ||
            ( lhs_p.state != rhs_p.state ) ||
            ( lhs_p.at_element != rhs_p.at_element ) ||
            ( lhs_p.at_turn != rhs_p.at_turn ) || ( lhs_p.id != rhs_p.id ) )
        {
            std::cout << "ERROR   :: difference in particle #" << ii << "\r\n"
                      << "        :: lhs particle       | "
                      <<         " rhs particle       | "
                      <<         " difference        \r\n"
                      << ".x      :: "
                      << std::setw( 20 ) << lhs_p.x << " | "
                      << std::setw( 20 ) << rhs_p.x << " | "
                      << std::setw( 20 ) << lhs_p.x - rhs_p.x << "\r\n"
                      << ".px     :: "
                      << std::setw( 20 ) << lhs_p.px << " | "
                      << std::setw( 20 ) << rhs_p.px << " | "
                      << std::setw( 20 ) << lhs_p.px - rhs_p.px << "\r\n"
                      << ".y      :: "
                      << std::setw( 20 ) << lhs_p.y << " | "
                      << std::setw( 20 ) << rhs_p.y << " | "
                      << std::setw( 20 ) << lhs_p.y - rhs_p.y << "\r\n"
                      << ".py     :: "
                      << std::setw( 20 ) << lhs_p.py << " | "
                      << std::setw( 20 ) << rhs_p.py << " | "
                      << std::setw( 20 ) << lhs_p.py - rhs_p.py << "\r\n"
                      << ".zeta   :: "
                      << std::setw( 20 ) << lhs_p.zeta << " | "
                      << std::setw( 20 ) << rhs_p.zeta << " | "
                      << std::setw( 20 ) << lhs_p.zeta - rhs_p.zeta << "\r\n"
                      << ".delta  :: "
                      << std::setw( 20 ) << lhs_p.delta << " | "
                      << std::setw( 20 ) << rhs_p.delta << " | "
                      << std::setw( 20 ) << lhs_p.delta - rhs_p.delta << "\r\n"
                      << ".rpp    :: "
                      << std::setw( 20 ) << lhs_p.rpp << " | "
                      << std::setw( 20 ) << rhs_p.rpp << " | "
                      << std::setw( 20 ) << lhs_p.rpp - rhs_p.rpp << "\r\n"
                      << ".rvv    :: "
                      << std::setw( 20 ) << lhs_p.rvv << " | "
                      << std::setw( 20 ) << rhs_p.rvv << " | "
                      << std::setw( 20 ) << lhs_p.rvv - rhs_p.rvv << "\r\n"
                      << ".psigma :: "
                      << std::setw( 20 ) << lhs_p.psigma << " | "
                      << std::setw( 20 ) << rhs_p.psigma << " | "
                      << std::setw( 20 ) << lhs_p.psigma - rhs_p.psigma << "\r\n"
                      << ".chi    :: "
                      << std::setw( 20 ) << lhs_p.chi << " | "
                      << std::setw( 20 ) << rhs_p.chi << " | "
                      << std::setw( 20 ) << lhs_p.chi - rhs_p.chi << "\r\n"
                      << ".qratio :: "
                      << std::setw( 20 ) << lhs_p.charge_ratio << " | "
                      << std::setw( 20 ) << rhs_p.charge_ratio << " | "
                      << std::setw( 20 ) << lhs_p.charge_ratio - rhs_p.charge_ratio << "\r\n"
                      << ".charge0:: "
                      << std::setw( 20 ) << lhs_p.charge0 << " | "
                      << std::setw( 20 ) << rhs_p.charge0 << " | "
                      << std::setw( 20 ) << lhs_p.charge0 - rhs_p.charge0 << "\r\n"
                      << ".mass0  :: "
                      << std::setw( 20 ) << lhs_p.mass0 << " | "
                      << std::setw( 20 ) << rhs_p.mass0 << " | "
                      << std::setw( 20 ) << lhs_p.mass0 - rhs_p.mass0 << "\r\n"
                      << ".beta0  :: "
                      << std::setw( 20 ) << lhs_p.beta0   << " | "
                      << std::setw( 20 ) << rhs_p.beta0   << " | "
                      << std::setw( 20 ) << lhs_p.beta0   - rhs_p.beta0   << "\r\n"
                      << ".gamma0 :: "
                      << std::setw( 20 ) << lhs_p.gamma0 << " | "
                      << std::setw( 20 ) << rhs_p.gamma0 << " | "
                      << std::setw( 20 ) << lhs_p.gamma0 - rhs_p.gamma0 << "\r\n"
                      << ".p0c    :: "
                      << std::setw( 20 ) << lhs_p.p0c << " | "
                      << std::setw( 20 ) << rhs_p.p0c << " | "
                      << std::setw( 20 ) << lhs_p.p0c - rhs_p.p0c << "\r\n"
                      << ".id     :: "
                      << std::setw( 20 ) << lhs_p.id << " | "
                      << std::setw( 20 ) << rhs_p.id << " | "
                      << std::setw( 20 ) << lhs_p.id - rhs_p.id << "\r\n"
                      << ".at_elem:: "
                      << std::setw( 20 ) << lhs_p.at_element << " | "
                      << std::setw( 20 ) << rhs_p.at_element << " | "
                      << std::setw( 20 ) << lhs_p.at_element - rhs_p.at_element << "\r\n"
                      << ".at_turn:: "
                      << std::setw( 20 ) << lhs_p.at_turn << " | "
                      << std::setw( 20 ) << rhs_p.at_turn << " | "
                      << std::setw( 20 ) << lhs_p.at_turn - rhs_p.at_turn << "\r\n"
                      << ".state  :: "
                      << std::setw( 20 ) << lhs_p.state << " | "
                      << std::setw( 20 ) << rhs_p.state << " | "
                      << std::setw( 20 ) << lhs_p.state - rhs_p.state
                      << "\r\n\r\n" << std::endl;
            success = false;
            if( stop_after_first_diff ) break;
        }
    }

    if( success )
    {
        std::cout << "Files " << path_to_lhs << " and \r\n"
                  << "  and " << path_to_rhs << " compare equivalent with \r\n"
                  << "abs_tol=" << abs_tol << " and rel_tol=" << rel_tol
                  << "\r\n" << std::endl;
    }

    return ( success ) ? 0 : 1;
}

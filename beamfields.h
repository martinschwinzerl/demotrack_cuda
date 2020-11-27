#ifndef DEMOTRACK_CUDA_BEAM_FIELDS_H__
#define DEMOTRACK_CUDA_BEAM_FIELDS_H__

#include <cmath>

#include "definitions.h"
#include "particle.h"

namespace demotrack
{
    struct __attribute__(( aligned( 8 ) )) DoublePair
    {
        double x;
        double y;
    };

    struct __attribute__((aligned( 8 ))) SpaceChargeCoasting
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 10 };
        }

        DEMOTRACK_FN static DoublePair TRANSVERSAL_FIELD_GAUSS_ROUND(
            double const sigma, double const delta_x, double const delta_y,
            double const x, double const y ) noexcept {

            using std::sqrt;
            using std::exp;

            static double const PI = double{ 3.141592653589793 };
            static double const EPSILON_0 = double{ 8.854187817620e-12 };
            static double const R_SQU_LIMIT = double{ 1e-10 };

            double const xx = x - delta_x;
            double const yy = y - delta_y;
            double const r_squ = xx * xx + yy * yy;

            double const temp = ( r_squ < R_SQU_LIMIT )
                ? sqrt( r_squ ) / ( 2. * PI * EPSILON_0 * sigma ) //linearised
                : ( 1. / ( 2. * PI * EPSILON_0 * r_squ ) ) *
                  ( 1. - exp( -0.5 * r_squ / ( sigma * sigma ) ) );

            return DoublePair{ temp * xx, temp * yy };
        }

        DEMOTRACK_FN static DoublePair CERRF(
            double const in_real, double const in_imag ) noexcept {

            using std::fabs;
            using std::pow;
            using std::sin;
            using std::cos;

            /* This function calculates the double precision complex error fnct.
            based on the algorithm of the FORTRAN function written at CERN by K. Koelbig
            Program C335, 1970. See also M. Bassetti and G.A. Erskine, "Closed
            expression for the electric field of a two-dimensional Gaussian charge
            density", CERN-ISR-TH/80-06; */

            DoublePair W = { 0., 0. };

            double const xLim = double{ 5.33 };
            double const yLim = double{ 4.29 };
            double const a_constant = double{ 1.12837916709551 };

            double const x = fabs( in_real );
            double const y = fabs( in_imag );

            double Rx[ 33 ] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0. };

            double Ry[ 33 ] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0. };

            if( ( y < yLim ) && ( x < xLim ) )
            {
                double const xx = x / xLim;
                double const q = ( 1. - y / yLim ) * sqrt( 1.0 - xx * xx );
                double const h  = 1. / ( 3.2 * q );
                int const nc = 7 + ( int )( 23. * q );

                double xl = pow( h, ( double )( 1 - nc ) );
                double const xh = y + 0.5 / h;
                double const yh = x;
                int const nu = 10 + ( int )( 21. * q );

                for( int n = nu; n > 0; n-- ){
                    double const Tx = xh + n * Rx[ n ];
                    double const Ty = yh - n * Ry[ n ];
                    double const Tn = Tx * Tx + Ty * Ty;
                    Rx[ n-1 ] = 0.5 * Tx / Tn;
                    Ry[ n-1 ] = 0.5 * Ty / Tn;
                }
                /* .... */
                double Sx = 0.;
                double Sy = 0.;

                for( int n = nc ; n > 0 ; n-- ){
                    double const Saux = Sx + xl;
                    Sx = Rx[ n - 1 ] * Saux - Ry[ n - 1 ] * Sy;
                    Sy = Rx[ n - 1 ] * Sy + Ry[ n - 1 ] * Saux;
                    xl = h * xl;
                };
                W.x = a_constant * Sx;
                W.y = a_constant * Sy;
            }
            else
            {
                double const xh = y;
                double const yh = x;

                for( int n = 9 ; n > 0 ; n-- )
                {
                    double const Tx = xh + n * Rx[ 0 ];
                    double const Ty = yh - n * Ry[ 0 ];
                    double const Tn = Tx * Tx + Ty * Ty;
                    Rx[ 0 ] = 0.5 * Tx / Tn;
                    Ry[ 0 ] = 0.5 * Ty / Tn;
                };

                W.x = a_constant * Rx[ 0 ];
                W.y = a_constant * Ry[ 0 ];
            }

            if( y == 0.){ W.x = exp( -x * x ); }
            if( in_imag < 0. )
            {
                W.x =   2. * exp( y * y - x * x ) * cos( 2. * x * y ) - W.x;
                W.y = - 2. * exp( y * y - x * x ) * sin( 2. * x * y ) - W.y;
                if( in_real > 0. ) { W.y = -W.y; }
            }
            else if( in_real < 0. )
            {
                W.y = -W.y;
            }

            return W;
        }

        DEMOTRACK_FN static DoublePair TRANSVERSAL_FIELD_GAUSS_ELLIPSE(
            double const sigma_x, double const sigma_y,
            double const delta_x, double const delta_y,
            double const x, double const y ) noexcept {

            using std::sqrt;
            using std::exp;
            using this_type = SpaceChargeCoasting;

            DoublePair E = { 0., 0. };

            double const xx = x - delta_x;
            double const sign_xx = ( ( double )( 0. < xx ) ) -
                                   ( ( double )( xx < 0. ) );

            double const yy = y - delta_y;
            double const sign_yy = ( ( double )( 0. < yy ) ) -
                                   ( ( double )( yy < 0. ) );

            double const abx = xx * sign_xx;
            double const aby = yy * sign_yy;

            double const SQRT_PI = double{ 1.7724538509055160273 };
            double const EPSILON_0 = double{ 8.854187817620e-12 };

            double const sigma_x_squ = sigma_x * sigma_x;
            double const sigma_y_squ = sigma_y * sigma_y;
            double const exp_be = exp( - abx * abx / ( 2.0 * sigma_x_squ )
                                       - aby * aby / ( 2.0 * sigma_y_squ ) );

            if( sigma_x > sigma_y )
            {
                double const S = sqrt( 2.* ( sigma_x_squ - sigma_y_squ ) );
                double const fact_be = 1. / ( 2. * EPSILON_0 * SQRT_PI * S );

                DoublePair const w_zeta = this_type::CERRF( abx / S, aby / S );
                DoublePair const w_eta  = this_type::CERRF(
                    ( abx * sigma_y ) / ( sigma_x * S ),
                    ( aby * sigma_x ) / ( sigma_y * S ) );

                E.x = fact_be * ( w_zeta.y - w_eta.y * exp_be );
                E.y = fact_be * ( w_zeta.x - w_eta.x * exp_be );
            }
            else if( sigma_x < sigma_y )
            {
                double const S = sqrt( 2. * ( sigma_y_squ - sigma_x_squ ) );
                double const fact_be = 1. / ( 2. * EPSILON_0 * SQRT_PI * S );

                DoublePair const w_zeta = this_type::CERRF( aby / S, abx / S );
                DoublePair const w_eta  = this_type::CERRF(
                    ( aby * sigma_x ) / ( sigma_y * S ),
                    ( abx * sigma_y ) / ( sigma_x * S ) );

                E.y = fact_be * ( w_zeta.y - w_eta.y * exp_be );
                E.x = fact_be * ( w_zeta.x - w_eta.x * exp_be );
            }

            E.x *= sign_xx;
            E.y *= sign_yy;

            return E;
        }

        DEMOTRACK_FN static void GAUSS_FIELD_COMPONENTS(
            double const x, double const y,
            double const sigma_x, double const sigma_y,
            double const min_sigma_diff,
            DoublePair& __restrict__ E,
            DoublePair* __restrict__ ptr_G ) noexcept
        {
            using this_type = SpaceChargeCoasting;

            double const PI = double{ 3.141592653589793 };
            double const EPSILON_0 = double{ 8.854187817620e-12 };

            double const d_sigma = sigma_x - sigma_y;
            double const d_sigma_sign = ( ( double )( 0. < d_sigma ) ) -
                                        ( ( double )( d_sigma < 0. ) );

            double const abs_d_sigma = d_sigma * d_sigma_sign;

            if( abs_d_sigma < min_sigma_diff )
            {
                double const avg_sigma = 0.5 * ( sigma_x + sigma_y );
                E = this_type::TRANSVERSAL_FIELD_GAUSS_ROUND(
                    avg_sigma, 0., 0., x, y );

                if( ptr_G != nullptr )
                {
                    double const x_squ = x * x;
                    double const y_squ = y * y;
                    double const r_squ = x_squ + y_squ;

                    double const fact_a = 1. / ( 2. * r_squ );
                    double const fact_b = 2. * avg_sigma * avg_sigma;
                    double const fact_c = PI * EPSILON_0 * fact_b;
                    double const exp_fact = exp( -( x_squ + y_squ ) / fact_b );

                    ptr_G->x = fact_a * (
                        y * E.y - x * E.x + ( x_squ * exp_fact ) / fact_c );

                    ptr_G->y = fact_a * ( x * E.x - y * E.y +
                        x * E.x - y * E.y + ( y_squ * exp_fact ) / fact_c );
                }
            }
            else
            {
                E = this_type::TRANSVERSAL_FIELD_GAUSS_ELLIPSE(
                    sigma_x, sigma_y, 0., 0., x, y );

                if( ptr_G != nullptr )
                {
                    double const sig_11 = sigma_x * sigma_x;
                    double const sig_33 = sigma_y * sigma_y;
                    double const fact_a = 1. / ( 2. * ( sig_11 - sig_33 ) );
                    double const fact_b = 2. * PI * EPSILON_0;
                    double const exp_fact = exp( -( x * x ) / ( 2. * sig_11 )
                                                 -( y * y ) / ( 2. * sig_33 ) );

                    ptr_G->x = -fact_a * ( x * E.x + y * E.y +
                        ( ( exp_fact * sigma_x ) / sigma_y - 1. ) / fact_b );

                    ptr_G->y = +fact_a * ( x * E.x + y * E.y +
                        ( ( exp_fact * sigma_y ) / sigma_x - 1. ) / fact_b );
                }
            }
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept{

            using this_type = SpaceChargeCoasting;
            DoublePair E = { 0., 0. };

            double const EV_TO_COULOMB = double{ 1.602176634e-19 };
            double const charge = EV_TO_COULOMB * p.charge0;
            double const p0c = EV_TO_COULOMB * p.p0c;

            double fact_kick = this->number_of_particles *
                this->length * p.chi * p.charge_ratio * charge * charge *
                ( 1. - p.beta0 * p.beta0 );

            fact_kick /= this->circumference * p0c * p.beta0 * p.rvv;

            this_type::GAUSS_FIELD_COMPONENTS(
                ( p.x - this->x_co ), ( p.y - this->y_co ),
                this->sigma_x, this->sigma_y, this->min_sigma_diff,
                E, nullptr );

            p.px += fact_kick * E.x;
            p.py += fact_kick * E.y;

            return slot_index + this_type::NUM_SLOTS();
        }

        double type_id;
        double number_of_particles;
        double circumference;
        double sigma_x;
        double sigma_y;
        double length;
        double x_co;
        double y_co;
        double min_sigma_diff;
        double enabled;
    };

}

#endif /* DEMOTRACK_CUDA_BEAM_FIELDS_H__ */

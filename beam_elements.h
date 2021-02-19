#ifndef DEMOTRACK_CUDA_BEAM_ELEMENTS_H__
#define DEMOTRACK_CUDA_BEAM_ELEMENTS_H__

#include <cmath>
#include <cuda.h>

#include "definitions.h"
#include "particle.h"

namespace demotrack
{
    typedef enum
    {
        BEAM_ELEMENT_DRIFT = 2,
        BEAM_ELEMENT_DRIFT_EXACT = 3,
        BEAM_ELEMENT_MULTIPOLE = 4,
        BEAM_ELEMENT_CAVITY = 5,
        BEAM_ELEMENT_XY_SHIFT = 6,
        BEAM_ELEMENT_SROTATION = 7,
        BEAM_ELEMENT_LIMIT_RECT = 11,
        BEAM_ELEMENT_LIMIT_ELLIPSE = 12,
        BEAM_ELEMENT_LIMIT_RECT_ELLIPSE = 13,
        BEAM_ELEMENT_SC_COASTING = 16,
        BEAM_ELEMENT_DIPEDGE = 24,
        BEAM_ELEMENT_END_OF_TRACK = 255
    }
    beam_element_type;

    struct __attribute__((aligned( 8 ))) Drift
    {
        static DEMOTRACK_FN constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 2 };
        }

        static DEMOTRACK_FN void GLOBAL_APERTURE_CHECK(
            Particle& __restrict__ p ) noexcept {

            double const sign_x = ( ( double )( 0. < p.x ) ) -
                                  ( ( double )( p.x < 0. ) );

            double const sign_y = ( ( double )( 0. < p.y ) ) -
                                  ( ( double )( p.y < 0. ) );

            /* Check whether |p.x| < 1.0 and |p.y| < 1.0 */
            p.state &= ( uint64_type )( ( sign_x * p.x < 1. ) &
                                        ( sign_y * p.y < 1. ) );

            // Effectively only increment if particle is not lost
            p.at_element += p.state;
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            using std::sqrt;

            double const xp = p.px * p.rpp;
            double const yp = p.py * p.rpp;

            p.x    += this->length * xp;
            p.y    += this->length * yp;
            p.zeta += this->length * ( p.rvv -
                ( double{ 1.0 } + double{ 0.5 } * ( xp * xp + yp * yp ) ) );

            /* NOTE: we do not increment p.at_element here -> this is done in
             * GLOBAL_APERTURE_CHECK */

            return slot_index + Drift::NUM_SLOTS();
        }

        double type_id;
        double length;
    };

    struct __attribute__((aligned( 8 ))) DriftExact
    {
        static DEMOTRACK_FN constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 2 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            using std::sqrt;

            double const delta_plus_one = p.delta + double{ 1.0 };
            double const lpzi = this->length / sqrt( delta_plus_one *
                delta_plus_one - ( p.px * p.px + p.py * p.py ) );

            p.x += p.px * lpzi;
            p.y += p.py * lpzi;
            p.zeta += p.rvv * this->length - delta_plus_one * lpzi;

            /* NOTE: we do not increment p.at_element here -> this is done in
             * GLOBAL_APERTURE_CHECK */

            return slot_index + DriftExact::NUM_SLOTS();
        }

        double type_id;
        double length;
    };

    struct __attribute__((aligned( 8 ))) Multipole
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 39 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            double dPx = double{ 0.0 };
            double dPy = double{ 0.0 };

            int64_type index_x = 2 * ( int64_type )this->order;

            for( ; index_x >= int64_type{ 0 } ; index_x -= 2 )
            {
                double const dPx_saved = dPx;
                dPx = this->bal[ index_x     ] + dPx       * p.x - dPy * p.y;
                dPy = this->bal[ index_x + 1 ] + dPx_saved * p.y + dPy * p.x;
            }

            dPx *= -p.chi;
            dPy *=  p.chi;

            if( ( this->hxl > 0. ) || ( this->hxl < 0. ) ||
                ( this->hyl > 0. ) || ( this->hyl < 0. ) )
            {
                double const hxl_x = this->hxl * p.x;
                double const hyl_y = this->hyl * p.y;

                p.zeta += p.chi * ( hyl_y - hxl_x );
                dPx += this->hxl + this->hxl * p.delta;
                dPy -= this->hyl + this->hyl * p.delta;

                if( this->length > 0. )
                {
                    dPx -= ( p.chi * this->bal[ 0 ] * hxl_x ) / this->length;
                    dPy += ( p.chi * this->bal[ 1 ] * hyl_y ) / this->length;
                }
            }

            p.px += dPx;
            p.py += dPy;

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + Multipole::NUM_SLOTS();
        }

        double type_id;
        double order;
        double length;
        double hxl;
        double hyl;
        double bal[ 34 ];
    };

    struct __attribute__((aligned( 8 ))) Cavity
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 4 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            using std::sin;

            double const PI = double{ 3.141592653589793 };
            double const C0 = double{ 299792458. };
            double const beta = p.beta0 * p.rvv;
            double const phase = ( PI / 180. ) * this->lag -
                ( ( 2. * PI ) / C0 ) * ( p.zeta / beta ) * this->frequency;

            double const charge = p.charge0 * p.charge_ratio;
            p.add_to_energy( this->voltage * charge * sin( phase ) );

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + Cavity::NUM_SLOTS();
        }

        double type_id;
        double voltage;
        double frequency;
        double lag;
    };

    struct __attribute__((aligned(8))) XYShift
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 3 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            p.x -= this->dx;
            p.y -= this->dy;

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + XYShift::NUM_SLOTS();
        }

        double type_id;
        double dx;
        double dy;
    };

    struct __attribute__((aligned(8))) SRotation
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 3 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            double const x_hat  = p.x  * this->cos_z + p.y  * this->sin_z;
            double const px_hat = p.px * this->cos_z + p.py * this->sin_z;

            p.y  = -p.x  * this->sin_z + p.y  * this->cos_z;
            p.py = -p.px * this->sin_z + p.py * this->cos_z;

            p.x  = x_hat;
            p.px = px_hat;

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + SRotation::NUM_SLOTS();
        }

        double type_id;
        double cos_z;
        double sin_z;
    };

    struct __attribute__((aligned( 8 ))) LimitRect
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 5 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            p.state &= ( ( p.x >= this->min_x ) && ( p.x <= this->max_x ) &&
                         ( p.y >= this->min_y ) && ( p.y <= this->max_y ) );

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + LimitRect::NUM_SLOTS();
        }

        double type_id;
        double min_x;
        double max_x;
        double min_y;
        double max_y;
    };

    struct __attribute__((aligned( 8 ))) LimitEllipse
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 4 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            p.state &= ( this->ab_squ > (
                p.x * p.x * this->b_squ + p.y * p.y * this->a_squ ) );

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + LimitEllipse::NUM_SLOTS();
        }

        double type_id;
        double a_squ;
        double b_squ;
        double ab_squ;
    };

    struct __attribute__((aligned( 8 ))) LimitRectEllipse
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 6 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            p.state &= (
                ( p.x <= this->max_x ) && ( p.x >= -this->max_x ) &&
                ( p.y <= this->max_y ) && ( p.y >= -this->max_y ) &&
                ( this->ab_squ > ( p.x * p.x * this->b_squ +
                                   p.y * p.y * this->a_squ ) ) );

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + LimitRectEllipse::NUM_SLOTS();
        }

        double type_id;
        double max_x;
        double max_y;
        double a_squ;
        double b_squ;
        double ab_squ;
    };

    struct __attribute__((aligned( 8 ))) DipoleEdge
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 3 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            p.px += this->r21 * p.x;
            p.py += this->r43 * p.y;

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            return slot_index + DipoleEdge::NUM_SLOTS();
        }

        double type_id;
        double r21;
        double r43;
    };

    struct __attribute__((aligned( 8 ))) EndOfTrack
    {
        DEMOTRACK_FN static constexpr uint64_type NUM_SLOTS() noexcept
        {
            return uint64_type{ 4 };
        }

        DEMOTRACK_FN uint64_type track( Particle& __restrict__ p,
            uint64_type const slot_index ) const noexcept
        {
            uint64_t const stored_slot_idx = ( uint64_type )this->next_slot_idx;
            uint64_type const min_next_slot_idex =
                slot_index + EndOfTrack::NUM_SLOTS();

            // NOTE: The particle should not be here if it is lost!!!
            // But: effectively only increment if particle is not lost
            p.at_element += p.state;

            if( ends_turn > 0. )
            {
                p.at_element = ( uint64_type )this->start_turn_at_element;
                ++p.at_turn;
            }

            return ( ( ends_turn > 0 ) || ( stored_slot_idx < slot_index ) ||
                     ( stored_slot_idx >= min_next_slot_idex ) )
                ? stored_slot_idx : min_next_slot_idex;
        }

        double type_id;
        double start_turn_at_element;
        double next_slot_idx;
        double ends_turn;
    };
}

#endif /* DEMOTRACK_CUDA_BEAM_ELEMENTS_H__ */

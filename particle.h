#ifndef DEMOTRACK_CUDA_PARTICLE_H__
#define DEMOTRACK_CUDA_PARTICLE_H__

#include <cmath>
#include "definitions.h"

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

}

#endif /* DEMOTRACK_CUDA_PARTICLE_H__ */

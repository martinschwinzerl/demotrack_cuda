#ifndef DEMOTRACK_CUDA_FODO_LATTICE_H__
#define DEMOTRACK_CUDA_FODO_LATTICE_H__

#include <algorithm>

#include "definitions.h"
#include "beam_elements.h"

namespace demotrack
{
    DEMOTRACK_HOST_FN uint64_type create_fodo_lattice(
        double* __restrict__ buffer,
        uint64_type const max_buffer_capacity,
        bool const add_end_of_turn_marker = false ) noexcept {

        namespace dt = demotrack;

        uint64_type allocated_num_slots = uint64_type{ 0 };
        uint64_type const required_slots =
            ( add_end_of_turn_marker ) ? 52u : 48u;

        if( ( buffer != nullptr ) && ( max_buffer_capacity >= required_slots ) )
        {
            std::fill( buffer, buffer + max_buffer_capacity, double{ 0. } );

            double const DRIFT_TYPE =
                static_cast< double >( dt::BEAM_ELEMENT_DRIFT );

            double const MULTIPOLE_TYPE =
                static_cast< double >( dt::BEAM_ELEMENT_MULTIPOLE );

            double const CAVITY_TYPE =
                static_cast< double >( dt::BEAM_ELEMENT_CAVITY );

            // ---------------------------------------------------------------------
            // element 0: multipole dipole0 ; starts at idx = 0
            buffer[  0 ] = MULTIPOLE_TYPE; // dipole0 : type_id
            buffer[  1 ] = 0.;             // dipole0 : order  == 0
            buffer[  2 ] = 1.;             // dipole0 : length == 1 m
            buffer[  3 ] = 0.1570796327;   // dipole0 : hxl == 0.1570796327
            buffer[  4 ] = 0.0;            // dipole0 : hyl == 0.0
            buffer[  5 ] = 0.1570796327;   // dipole0 : bal[ 0 ] = 0.1570796327
            // ---------------------------------------------------------------------
            // element 1: drift drift0 ;   starts at idx = 9
            buffer[  9 ] = DRIFT_TYPE;     // drift0 : type_id
            buffer[ 10 ] = 5.;             // drift0 : length == 5 m
            // ---------------------------------------------------------------------
            // element 2: multipole q0 ;   starts at idx = 11
            buffer[ 11 ] = MULTIPOLE_TYPE; // q0 : type_id
            buffer[ 12 ] = 1.0;            // q0 : order  == 1
            buffer[ 13 ] = 1.0;            // q0 : length == 0 m
            buffer[ 14 ] = 0.0;            // q0 : hxl == 0.0
            buffer[ 15 ] = 0.0;            // q0 : hyl == 0.0
            buffer[ 16 ] = 0.0;            // q0 : bal[ 0 ] = 0.0
            buffer[ 17 ] = 0.0;            // q0 : bal[ 1 ] = 0.0
            buffer[ 18 ] = 0.1657145946;   // q0 : bal[ 2 ] = 0.1657145946
            buffer[ 19 ] = 0.0;            // q0 : bal[ 3 ] = 0.0
            // ---------------------------------------------------------------------
            // element 3: drift drift1 ;   starts at idx = 20
            buffer[ 20 ] = DRIFT_TYPE;     // drift1 : type_id
            buffer[ 21 ] = 5.0;            // drift1 : length == 5 m
            // ---------------------------------------------------------------------
            // element 4: multipole dipole1 ; starts at idx = 22
            buffer[ 22 ] = MULTIPOLE_TYPE; // dipole1 : type_id
            buffer[ 23 ] = 0.0;            // dipole1 : order  == 0
            buffer[ 24 ] = 1.0;            // dipole1 : length == 1 m
            buffer[ 25 ] = 0.1570796327;   // dipole1 : hxl == 0.1570796327
            buffer[ 26 ] = 0.0;            // dipole1 : hyl == 0.0
            buffer[ 27 ] = 0.1570796327;   // dipole1 : bal[ 0 ] = 0.1570796327
            // ---------------------------------------------------------------------
            // element 5: drift drift2 ;   starts at idx = 31
            buffer[ 31 ] = DRIFT_TYPE;     // drift2 : type_id
            buffer[ 32 ] = 5.0;            // drift2 : length == 5 m
            // ---------------------------------------------------------------------
            // element 6: multipole q1 ;   starts at idx = 33
            buffer[ 33 ] = MULTIPOLE_TYPE; // q1 : type_id
            buffer[ 34 ] = 1.0;            // q1 : order  == 1
            buffer[ 35 ] = 1.0;            // q1 : length == 0 m
            buffer[ 36 ] = 0.0;            // q1 : hxl == 0.0
            buffer[ 37 ] = 0.0;            // q1 : hyl == 0.0
            buffer[ 38 ] = 0.0;            // q1 : bal[ 0 ] = 0.0
            buffer[ 39 ] = 0.0;            // q1 : bal[ 1 ] = 0.0
            buffer[ 40 ] = -0.1657145946;  // q1 : bal[ 2 ] = -0.1685973315
            buffer[ 41 ] = 0.0;            // q1 : bal[ 3 ] = 0.0
            // ---------------------------------------------------------------------
            // element 7: drift drift3 ;   starts at idx = 42
            buffer[ 42 ] = DRIFT_TYPE;     // drift3 : type_id
            buffer[ 43 ] = 5.;             // drift3 : length == 5 m
            // ---------------------------------------------------------------------
            // element 8: cavity cavity0 ; starts at idx = 44
            buffer[ 44 ] = CAVITY_TYPE;    // drift3 : type_id
            buffer[ 45 ] = 5000000.0;      // cavity0  : voltage == 5000000 V
            buffer[ 46 ] = 239833966.0;    // cavity0  : frequency == 239833966 Hz
            buffer[ 47 ] = 180.0;          // cavity0   : lag == 180 degrees

            if( add_end_of_turn_marker )
            {
                // ---------------------------------------------------------------------
                // element 9: end-of-turn marker eot0; starts at idx = 48
                buffer[ 48 ] = static_cast< double >(
                    dt::BEAM_ELEMENT_END_OF_TRACK );

                buffer[ 49 ] = 0.0;    // eot0 : start_turn_at_element == 0
                buffer[ 50 ] = 0.0;    // eot0 : next_slot_idx == 0
                buffer[ 51 ] = 1.0;    // eot0 : ends_turn == 1
            }

            allocated_num_slots = required_slots;
        }

        return allocated_num_slots;
    }
}

#endif /* DEMOTRACK_CUDA_FODO_LATTICE_H__ */

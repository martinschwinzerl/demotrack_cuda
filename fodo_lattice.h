#ifndef DEMOTRACK_CUDA_FODO_LATTICE_H__
#define DEMOTRACK_CUDA_FODO_LATTICE_H__

#include <algorithm>

#include "definitions.h"
#include "beam_elements.h"

namespace demotrack
{
    typedef enum : uint64_type
    {
        FODO_LATTICE_USE_DRIFT_EXACT_FLAG = 0x01,
        FODO_LATTICE_ADD_END_TRACK_MARKER = 0x02
    }
    create_fodo_lattice_flags_type;

    DEMOTRACK_HOST_FN uint64_type create_fodo_lattice(
        double* __restrict__ buffer, uint64_type const max_buffer_capacity,
        uint64_type const flags = uint64_type{ 0 } ) noexcept
    {
        namespace dt = demotrack;

        uint64_type allocated_num_slots = uint64_type{ 0 };

        bool const add_end_of_turn_marker = (
            dt::FODO_LATTICE_ADD_END_TRACK_MARKER ==
                ( flags & dt::FODO_LATTICE_ADD_END_TRACK_MARKER ) );

        uint64_type const required_slots = ( add_end_of_turn_marker )
            ? 172u : 168u;

        if( ( buffer != nullptr ) && ( max_buffer_capacity >= required_slots ) )
        {
            std::fill( buffer, buffer + max_buffer_capacity, double{ 0. } );

            double const DRIFT_TYPE = ( dt::FODO_LATTICE_USE_DRIFT_EXACT_FLAG ==
                ( dt::FODO_LATTICE_USE_DRIFT_EXACT_FLAG & flags ) )
                ? static_cast< double >( dt::BEAM_ELEMENT_DRIFT_EXACT )
                : static_cast< double >( dt::BEAM_ELEMENT_DRIFT );

            double const MULTIPOLE_TYPE =
                static_cast< double >( dt::BEAM_ELEMENT_MULTIPOLE );

            double const CAVITY_TYPE =
                static_cast< double >( dt::BEAM_ELEMENT_CAVITY );

            // ---------------------------------------------------------------------
            // element 0: multipole dipole0 ; starts at idx = 0
            buffer[   0 ] = MULTIPOLE_TYPE; // dipole0 : type_id
            buffer[   1 ] = 0.;             // dipole0 : order  == 0
            buffer[   2 ] = 1.;             // dipole0 : length == 1 m
            buffer[   3 ] = 0.1570796327;   // dipole0 : hxl == 0.1570796327
            buffer[   4 ] = 0.0;            // dipole0 : hyl == 0.0
            buffer[   5 ] = 0.1570796327;   // dipole0 : bal[ 0 ] = 0.1570796327
            // ---------------------------------------------------------------------
            // element 1: drift drift0 ;   starts at idx = 39
            buffer[  39 ] = DRIFT_TYPE;     // drift0 : type_id
            buffer[  40 ] = 5.;             // drift0 : length == 5 m
            // ---------------------------------------------------------------------
            // element 2: multipole q0 ;   starts at idx = 41
            buffer[  41 ] = MULTIPOLE_TYPE; // q0 : type_id
            buffer[  42 ] = 1.0;            // q0 : order  == 1
            buffer[  43 ] = 1.0;            // q0 : length == 0 m
            buffer[  44 ] = 0.0;            // q0 : hxl == 0.0
            buffer[  45 ] = 0.0;            // q0 : hyl == 0.0
            buffer[  46 ] = 0.0;            // q0 : bal[ 0 ] = 0.0
            buffer[  47 ] = 0.0;            // q0 : bal[ 1 ] = 0.0
            buffer[  48 ] = 0.1657145946;   // q0 : bal[ 2 ] = 0.1657145946
            buffer[  49 ] = 0.0;            // q0 : bal[ 3 ] = 0.0
            // ---------------------------------------------------------------------
            // element 3: drift drift1 ;   starts at idx = 80
            buffer[  80 ] = DRIFT_TYPE;     // drift1 : type_id
            buffer[  81 ] = 5.0;            // drift1 : length == 5 m
            // ---------------------------------------------------------------------
            // element 4: multipole dipole1 ; starts at idx = 82
            buffer[  82 ] = MULTIPOLE_TYPE; // dipole1 : type_id
            buffer[  83 ] = 0.0;            // dipole1 : order  == 0
            buffer[  84 ] = 1.0;            // dipole1 : length == 1 m
            buffer[  85 ] = 0.1570796327;   // dipole1 : hxl == 0.1570796327
            buffer[  86 ] = 0.0;            // dipole1 : hyl == 0.0
            buffer[  87 ] = 0.1570796327;   // dipole1 : bal[ 0 ] = 0.1570796327
            // ---------------------------------------------------------------------
            // element 5: drift drift2 ;   starts at idx = 121
            buffer[ 121 ] = DRIFT_TYPE;     // drift2 : type_id
            buffer[ 122 ] = 5.0;            // drift2 : length == 5 m
            // ---------------------------------------------------------------------
            // element 6: multipole q1 ;   starts at idx = 123
            buffer[ 123 ] = MULTIPOLE_TYPE; // q1 : type_id
            buffer[ 124 ] = 1.0;            // q1 : order  == 1
            buffer[ 125 ] = 1.0;            // q1 : length == 0 m
            buffer[ 126 ] = 0.0;            // q1 : hxl == 0.0
            buffer[ 127 ] = 0.0;            // q1 : hyl == 0.0
            buffer[ 128 ] = 0.0;            // q1 : bal[ 0 ] = 0.0
            buffer[ 129 ] = 0.0;            // q1 : bal[ 1 ] = 0.0
            buffer[ 130 ] = -0.1657145946;  // q1 : bal[ 2 ] = -0.1685973315
            buffer[ 131 ] = 0.0;            // q1 : bal[ 3 ] = 0.0
            // ---------------------------------------------------------------------
            // element 7: drift drift3 ;   starts at idx = 162
            buffer[ 162 ] = DRIFT_TYPE;     // drift3 : type_id
            buffer[ 163 ] = 5.;             // drift3 : length == 5 m
            // ---------------------------------------------------------------------
            // element 8: cavity cavity0 ; starts at idx = 164
            buffer[ 164 ] = CAVITY_TYPE;    // drift3 : type_id
            buffer[ 165 ] = 5000000.0;      // cavity0  : voltage == 5000000 V
            buffer[ 166 ] = 239833966.0;    // cavity0  : frequency == 239833966 Hz
            buffer[ 167 ] = 180.0;          // cavity0   : lag == 180 degrees

            if( add_end_of_turn_marker )
            {
                // ---------------------------------------------------------------------
                // element 9: end-of-turn marker eot0; starts at idx = 168
                buffer[ 168 ] = static_cast< double >(
                    dt::BEAM_ELEMENT_END_OF_TRACK );

                buffer[ 169 ] = 0.0;    // eot0 : start_turn_at_element == 0
                buffer[ 170 ] = 0.0;    // eot0 : next_slot_idx == 0
                buffer[ 171 ] = 1.0;    // eot0 : ends_turn == 1
            }

            allocated_num_slots = required_slots;
        }

        return allocated_num_slots;
    }
}

#endif /* DEMOTRACK_CUDA_FODO_LATTICE_H__ */

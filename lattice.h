#ifndef DEMOTRACK_CUDA_LATTICE_H__
#define DEMOTRACK_CUDA_LATTICE_H__

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "definitions.h"
#include "beam_elements.h"
#if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && ( DEMOTRACK_ENABLE_BEAMFIELDS == 1 )
#include "beamfields.h"
#endif /* BEAMFIELDS */

namespace demotrack
{
    typedef enum : uint64_type
    {
        FODO_LATTICE_FLAGS_NONE            = 0x00,
        FODO_LATTICE_USE_DRIFT_EXACT_FLAG  = 0x01,
        FODO_LATTICE_ADD_END_TRACK_MARKER  = 0x02,
        FODO_LATTICE_ADD_DUMMY_SPACECHARGE = 0x04
    }
    create_fodo_lattice_flags_type;

    DEMOTRACK_HOST_FN uint64_type create_fodo_lattice(
        double* __restrict__ buffer, uint64_type const max_buffer_capacity,
        uint64_type const flags = uint64_type{ 0 } ) noexcept {
        namespace dt = demotrack;
        uint64_type allocated_num_slots = uint64_type{ 0 };

        bool const use_drift_exact_flag = (
            dt::FODO_LATTICE_USE_DRIFT_EXACT_FLAG ==
                ( flags & dt::FODO_LATTICE_USE_DRIFT_EXACT_FLAG ) );

        bool const add_end_of_turn_marker = (
            dt::FODO_LATTICE_ADD_END_TRACK_MARKER ==
                ( flags & dt::FODO_LATTICE_ADD_END_TRACK_MARKER ) );

        #if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && \
                   ( DEMOTRACK_ENABLE_BEAMFIELDS == 1 )

        bool const add_dummy_spacecharge = (
            dt::FODO_LATTICE_ADD_DUMMY_SPACECHARGE ==
                ( flags & dt::FODO_LATTICE_ADD_DUMMY_SPACECHARGE ) );

        #endif /* DEMOTRACK_ENABLE_BEAMFIELDS */

        uint64_type required_slots = ( use_drift_exact_flag )
            ? uint64_type{ 4u } * dt::DriftExact::NUM_SLOTS()
            : uint64_type{ 4u } * dt::Drift::NUM_SLOTS();

        required_slots += uint64_type{ 4u } * dt::Multipole::NUM_SLOTS();
        required_slots += dt::Cavity::NUM_SLOTS();

        if( add_end_of_turn_marker ) {
            required_slots += dt::EndOfTrack::NUM_SLOTS(); }

        #if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && \
                   ( DEMOTRACK_ENABLE_BEAMFIELDS == 1 )

        if( add_dummy_spacecharge  ) {
            required_slots += dt::SpaceChargeCoasting::NUM_SLOTS(); }

        #endif /* DEMOTRACK_ENABLE_BEAMFIELDS */

        if( ( buffer != nullptr ) &&
            ( max_buffer_capacity > uint64_type{ 0 } ) )
        {
            std::fill( buffer, buffer + max_buffer_capacity, double{ 0. } );
        }

        if( ( buffer != nullptr ) && ( max_buffer_capacity >= required_slots ) )
        {
            uint64_type ii = uint64_type{ 0 };

            double const DRIFT_TYPE = ( use_drift_exact_flag )
                ? static_cast< double >( dt::BEAM_ELEMENT_DRIFT_EXACT )
                : static_cast< double >( dt::BEAM_ELEMENT_DRIFT );

            uint64_type const DRIFT_NUM_SLOTS = ( use_drift_exact_flag )
                ? dt::DriftExact::NUM_SLOTS() : dt::Drift::NUM_SLOTS();

            double const MULTIPOLE_TYPE = static_cast< double >(
                dt::BEAM_ELEMENT_MULTIPOLE );

            double const CAVITY_TYPE = static_cast< double >(
                dt::BEAM_ELEMENT_CAVITY );

            #if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && \
                       ( DEMOTRACK_ENABLE_BEAMFIELDS == 1 )

            double const SC_TYPE = static_cast< double >(
                dt::BEAM_ELEMENT_SC_COASTING );

            #endif /* DEMOTRACK_ENABLE_BEAMFIELDS */

            double const EOT_TYPE = static_cast< double >(
                dt::BEAM_ELEMENT_END_OF_TRACK );

            // ---------------------------------------------------------------------
            // element 0: multipole dipole0 ; starts at idx = 0
            buffer[ ii + 0 ] = MULTIPOLE_TYPE; // dipole0 : type_id
            buffer[ ii + 1 ] = 0.;             // dipole0 : order  == 0
            buffer[ ii + 2 ] = 1.;             // dipole0 : length == 1 m
            buffer[ ii + 3 ] = 0.1570796327;   // dipole0 : hxl == 0.1570796327
            buffer[ ii + 4 ] = 0.0;            // dipole0 : hyl == 0.0
            buffer[ ii + 5 ] = 0.1570796327;   // dipole0 : bal[ 0 ] = 0.1570796327
            ii += dt::Multipole::NUM_SLOTS();
            // ---------------------------------------------------------------------
            // element 1: drift drift0 ;   starts at idx = 39
            buffer[ ii + 0 ] = DRIFT_TYPE;     // drift0 : type_id
            buffer[ ii + 1 ] = 5.;             // drift0 : length == 5 m
            ii += DRIFT_NUM_SLOTS;
            // ---------------------------------------------------------------------
            // element 2: multipole q0 ;   starts at idx = 41
            buffer[ ii + 0 ] = MULTIPOLE_TYPE; // q0 : type_id
            buffer[ ii + 1 ] = 1.0;            // q0 : order  == 1
            buffer[ ii + 2 ] = 1.0;            // q0 : length == 0 m
            buffer[ ii + 3 ] = 0.0;            // q0 : hxl == 0.0
            buffer[ ii + 4 ] = 0.0;            // q0 : hyl == 0.0
            buffer[ ii + 5 ] = 0.0;            // q0 : bal[ 0 ] = 0.0
            buffer[ ii + 6 ] = 0.0;            // q0 : bal[ 1 ] = 0.0
            buffer[ ii + 7 ] = 0.1657145946;   // q0 : bal[ 2 ] = 0.1657145946
            buffer[ ii + 8 ] = 0.0;            // q0 : bal[ 3 ] = 0.0
            ii += dt::Multipole::NUM_SLOTS();
            // ---------------------------------------------------------------------
            // element 3: drift drift1 ;   starts at idx = 80
            buffer[ ii + 0 ] = DRIFT_TYPE;     // drift1 : type_id
            buffer[ ii + 1 ] = 5.0;            // drift1 : length == 5 m
            ii += DRIFT_NUM_SLOTS;
            // ---------------------------------------------------------------------
            // element 4: multipole dipole1 ; starts at idx = 82
            buffer[  ii + 0 ] = MULTIPOLE_TYPE; // dipole1 : type_id
            buffer[  ii + 1 ] = 0.0;            // dipole1 : order  == 0
            buffer[  ii + 2 ] = 1.0;            // dipole1 : length == 1 m
            buffer[  ii + 3 ] = 0.1570796327;   // dipole1 : hxl == 0.1570796327
            buffer[  ii + 4 ] = 0.0;            // dipole1 : hyl == 0.0
            buffer[  ii + 5 ] = 0.1570796327;   // dipole1 : bal[ 0 ] = 0.1570796327
            ii += dt::Multipole::NUM_SLOTS();
            // ---------------------------------------------------------------------
            // element 5: drift drift2 ;   starts at idx = 121
            buffer[ ii + 0 ] = DRIFT_TYPE;     // drift2 : type_id
            buffer[ ii + 1 ] = 5.0;            // drift2 : length == 5 m
            ii += DRIFT_NUM_SLOTS;
            // ---------------------------------------------------------------------
            // element 6: multipole q1 ;   starts at idx = 123
            buffer[ ii + 0 ] = MULTIPOLE_TYPE; // q1 : type_id
            buffer[ ii + 1 ] = 1.0;            // q1 : order  == 1
            buffer[ ii + 2 ] = 1.0;            // q1 : length == 0 m
            buffer[ ii + 3 ] = 0.0;            // q1 : hxl == 0.0
            buffer[ ii + 4 ] = 0.0;            // q1 : hyl == 0.0
            buffer[ ii + 5 ] = 0.0;            // q1 : bal[ 0 ] = 0.0
            buffer[ ii + 6 ] = 0.0;            // q1 : bal[ 1 ] = 0.0
            buffer[ ii + 7 ] = -0.1657145946;  // q1 : bal[ 2 ] = -0.1685973315
            buffer[ ii + 8 ] = 0.0;            // q1 : bal[ 3 ] = 0.0
            ii += dt::Multipole::NUM_SLOTS();
            // ---------------------------------------------------------------------
            // element 7: drift drift3 ;   starts at idx = 162
            buffer[ ii + 0 ] = DRIFT_TYPE;     // drift3 : type_id
            buffer[ ii + 1 ] = 5.;             // drift3 : length == 5 m
            ii += DRIFT_NUM_SLOTS;
            // ---------------------------------------------------------------------
            // element 8: cavity cavity0 ; starts at idx = 164
            buffer[ ii + 0 ] = CAVITY_TYPE;    // cavity0 : type_id
            buffer[ ii + 1 ] = 5000000.0;      // cavity0 : voltage == 5000000 V
            buffer[ ii + 2 ] = 239833966.0;    // cavity0 : frequency == 239833966 Hz
            buffer[ ii + 3 ] = 180.0;          // cavity0 : lag == 180 degrees
            ii += dt::Cavity::NUM_SLOTS();

            #if defined( DEMOTRACK_ENABLE_BEAMFIELDS ) && \
                       ( DEMOTRACK_ENABLE_BEAMFIELDS == 1 )

            if( add_dummy_spacecharge )
            {
                buffer[ ii + 0 ] = SC_TYPE;    // spacecharge0 : type_id
                buffer[ ii + 1 ] = 0.0;        // spacecharge0 : num_of_particles == 0
                buffer[ ii + 2 ] = 1.0;        // spacecharge0 : circumference == 1.0
                buffer[ ii + 3 ] = 1e-3;       // spacecharge0 : sigma_x == 1 mm
                buffer[ ii + 4 ] = 1e-3;       // spacecharge0 : sigma_y == 1 mm
                buffer[ ii + 5 ] = 0.0;        // spacecharge0 : length == 0.0 m
                buffer[ ii + 6 ] = 0.0;        // spacecharge0 : x_co == 0.0 m
                buffer[ ii + 7 ] = 0.0;        // spacecharge0 : y_co == 0.0 m
                buffer[ ii + 8 ] = 1e-6;       // spacecharge0 : min_sigma_diff
                buffer[ ii + 9 ] = 1.0;        // spacecharge0 : enabled == 1.0
                ii += dt::SpaceChargeCoasting::NUM_SLOTS();
            }

            #endif /* DEMOTRACK_ENABLE_BEAMFIELDS  */

            if( add_end_of_turn_marker )
            {
                // ---------------------------------------------------------------------
                // element 9: end-of-turn marker eot0; starts at idx = 168
                buffer[ ii + 0 ] = EOT_TYPE; // eot0 : type_id
                buffer[ ii + 1 ] = 0.0;      // eot0 : start_turn_at_element == 0
                buffer[ ii + 2 ] = 0.0;      // eot0 : next_slot_idx == 0
                buffer[ ii + 3 ] = 1.0;      // eot0 : ends_turn == 1
                ii += dt::EndOfTrack::NUM_SLOTS();
            }

            allocated_num_slots = ii;
        }

        return allocated_num_slots;
    }

    DEMOTRACK_HOST_FN uint64_type load_lattice(
        std::vector< double >& __restrict__ lattice_data,
        std::string& path_to_lattice_data,
        demotrack::create_fodo_lattice_flags_type const flags =
            demotrack::FODO_LATTICE_FLAGS_NONE ) {

        namespace dt = demotrack;
        dt::uint64_type lattice_size = dt::uint64_type{ 0 };
        lattice_data.clear();

        if( !path_to_lattice_data.empty() )
        {
            ::FILE* fp = std::fopen( path_to_lattice_data.c_str(), "rb" );

            if( fp != nullptr )
            {
                double temp;

                dt::uint64_type const ret = std::fread(
                    &temp, sizeof( double ), 1u, fp );

                dt::uint64_type const num_slots = static_cast<
                    dt::uint64_type >( temp );

                if( ( ret == 1u ) && ( num_slots > 0u ) )
                {
                    lattice_data.resize( num_slots, double{ 0.0 } );
                    dt::uint64_type const num_slots_read = std::fread(
                        lattice_data.data(), sizeof( double ), num_slots, fp );

                    if( num_slots_read == num_slots )
                    {
                        lattice_size = num_slots;
                    }
                }
            }
        }

        if( lattice_size == 0u )
        {
            path_to_lattice_data.clear();
            lattice_data.resize( 200u, double{ 0 } );
            lattice_size = dt::create_fodo_lattice(
                lattice_data.data(), 200u, flags );
        }

        return lattice_size;
    }
}

#endif /* DEMOTRACK_CUDA_FODO_LATTICE_H__ */

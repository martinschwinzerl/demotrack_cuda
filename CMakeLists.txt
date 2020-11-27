cmake_minimum_required( VERSION 3.11 FATAL_ERROR )
project( demotrack_cuda_cxx LANGUAGES C CXX )

# -------------------------------------------------------------------------------
# Enable CUDA support

include( CheckLanguage )
check_language( CUDA )

if( CMAKE_CUDA_COMPILER )
    message( STATUS "-- enabling CUDA language" )
    enable_language( CUDA )
    message( STATUS "-- CUDA compiler version ${CMAKE_CUDA_COMPILER_VERSION}" )
endif()

if( CMAKE_VERSION VERSION_GREATER_EQUAL 3.17.0 )
    find_package( CudaToolkit REQUIRED )
    set( DEMOTRACK_LINK_LIBRARIES ${DEMOTRACK_LINK_LIBRARIES} CUDA::cudart )
else()
    find_package( CUDA REQUIRED )
    set( DEMOTRACK_LINK_LIBRARIES ${DEMOTRACK_LINK_LIBRARIES} ${CUDA_LIBRARIES} )
endif()

# ******************************************************************************
# demo applications

set( DEMO_APPLICATIONS )
set( DEMO_APPLICATIONS_WITH_SC )
include_directories( ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------
# demo01: particles stored in global memory, lattice stored in global memory

add_executable( demo01_sc0 demo01.cu )
set( DEMO_APPLICATIONS ${DEMO_APPLICATIONS} demo01_sc0 )
target_link_libraries( demo01_sc0 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

add_executable( demo01_sc1 demo01.cu )
set( DEMO_APPLICATIONS ${DEMO_APPLICATIONS} demo01_sc1 )
set( DEMO_APPLICATIONS_WITH_SC ${DEMO_APPLICATIONS_WITH_SC} demo01_sc1 )
target_link_libraries( demo01_sc1 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )


# ------------------------------------------------------------------------------
# demo02: particles stored in global memory, lattice stored in global memory
#         BUT: tracking with thread-local / private copy of the particles,
#              i.e. avoiding the frequent global memory accesses

add_executable( demo02_sc0 demo02.cu )
set( DEMO_APPLICATIONS ${DEMO_APPLICATIONS} demo02_sc0 )
target_link_libraries( demo02_sc0 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

add_executable( demo02_sc1 demo02.cu )
set( DEMO_APPLICATIONS ${DEMO_APPLICATIONS} demo02_sc1 )
set( DEMO_APPLICATIONS_WITH_SC ${DEMO_APPLICATIONS_WITH_SC} demo02_sc1 )
target_link_libraries( demo02_sc1 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

# ==============================================================================
# Apply common properties to all TARGETs in DEMO_APPLICATIONS:

set_property( TARGET ${DEMO_APPLICATIONS} PROPERTY POSITION_INDEPENDENT_CODE ON )
set_property( TARGET ${DEMO_APPLICATIONS} PROPERTY CXX_STANDARD 11 )
set_property( TARGET ${DEMO_APPLICATIONS} PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET ${DEMO_APPLICATIONS} PROPERTY CUDA_STANDARD 11 )
set_property( TARGET ${DEMO_APPLICATIONS} PROPERTY CUDA_STANDARD_REQUIRED ON )

set( DEMOTRACK_COMPILE_FLAGS )

if( CMAKE_VERSION VERSION_GREATER_EQUAL 3.18.0 )
    set_property( TARGET ${DEMO_APPLICATIONS} PROPERTY CUDA_ARCHITECTURES 35 )
else()
    string( APPEND DEMOTRACK_COMPILE_FLAGS " -arch=sm_35" )
endif()

string( APPEND DEMOTRACK_COMPILE_FLAGS " -fmad=false" )
string( APPEND DEMOTRACK_COMPILE_FLAGS " -lineinfo" )
string( APPEND DEMOTRACK_COMPILE_FLAGS " -D_FORCE_INLINES" )

set_property( TARGET ${DEMO_APPLICATIONS} APPEND PROPERTY
    COMPILE_FLAGS ${DEMOTRACK_COMPILE_FLAGS} )

if( DEMO_APPLICATIONS_WITH_SC )
    set_property( TARGET ${DEMO_APPLICATIONS} APPEND PROPERTY
        COMPILE_OPTIONS  -DDEMOTRACK_ENABLE_BEAMFIELDS=1 )
endif()



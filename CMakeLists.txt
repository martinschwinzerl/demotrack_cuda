cmake_minimum_required( VERSION 3.11 FATAL_ERROR )
project( demotrack_cuda_cxx LANGUAGES C CXX )

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Build type [Debug ReleaseRelWithDebInfo MinSizeRel]" FORCE )
endif()

if( EXISTS ${CMAKE_SOURCE_DIR}/Settings.cmake )
    message( STATUS "loading custom settings file ${CMAKE_SOURCE_DIR}/Settings.cmake" )
    include( ${CMAKE_SOURCE_DIR}/Settings.cmake )
else()
    message( STATUS "loading default settings file ${CMAKE_SOURCE_DIR}/Settings.cmake.original" )
    include( ${CMAKE_SOURCE_DIR}/Settings.cmake.original )
endif()

if( USE_BLOCKSIZE_CALCULATOR )
    set( DEMOTRACK_USE_BLOCKSIZE_CALCULATOR "1" )
else()
    set( DEMOTRACK_USE_BLOCKSIZE_CALCULATOR "0" )
endif()

configure_file( config.h.template ${CMAKE_SOURCE_DIR}/config.h @ONLY )

# -------------------------------------------------------------------------------
# Enable CUDA support

# Set environment variable CUDA_BIN_PATH to the directory containing nvcc or
# CUDA_TOOLKIT_ROOT_DIR variable to hint the CUDA install location
find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIRS} )

# ******************************************************************************
# demo applications

set( CMAKE_POSITION_INDEPENDENT_CODE ON )
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CUDA_STANDARD 11 )
set( CMAKE_CUDA_STANDARD_REQUIRED ON )

include_directories( ${CMAKE_SOURCE_DIR} )

set( COMMON_HEADERS definitions.h beam_elements.h beamfields.h
                    lattice.h particle.h )

add_subdirectory( tools )

# ------------------------------------------------------------------------------
# demo01: particles stored in global memory, lattice stored in global memory

set( CUDA_NVCC_FLAGS "-arch=${DEMOTRACK_CUDA_SM_ARCH};-fmad=false;-D_FORCE_INLINES" )

if( DEFINED DEMOTRACK_CUDA_OPTIMIZATION AND
    NOT "${DEMOTRACK_CUDA_OPTIMIZATION}" STREQUAL "" )
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas=${DEMOTRACK_CUDA_OPTIMIZATION}" )
endif()

if( DEMOTRACK_CUDA_ENABLE_LINEINFO )
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo" )
endif()

CUDA_ADD_EXECUTABLE( demo01_sc0 demo01.cu ${COMMON_HEADERS} )
target_link_libraries( demo01_sc0 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

CUDA_ADD_EXECUTABLE( demo02_sc0 demo02.cu ${COMMON_HEADERS} )
target_link_libraries( demo02_sc0 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

# ------------------------------------------------------------------------------
# demo02: particles stored in global memory, lattice stored in global memory
#         BUT: tracking with thread-local / private copy of the particles,
#              i.e. avoiding the frequent global memory accesses

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DDEMOTRACK_ENABLE_BEAMFIELDS=1" )

CUDA_ADD_EXECUTABLE( demo01_sc1 demo01.cu ${COMMON_HEADERS} )
target_link_libraries( demo01_sc1 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

CUDA_ADD_EXECUTABLE( demo02_sc1 demo02.cu ${COMMON_HEADERS} )
target_link_libraries( demo02_sc1 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )


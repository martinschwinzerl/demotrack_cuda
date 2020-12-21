cmake_minimum_required( VERSION 3.11 FATAL_ERROR )
project( demotrack_cuda_cxx LANGUAGES C CXX )

# -------------------------------------------------------------------------------
# Enable CUDA support

# Set environment variable CUDA_BIN_PATH to the directory containing nvcc or
# CUDA_TOOLKIT_ROOT_DIR variable to hint the CUDA install location
find_package( CUDA REQUIRED )

# ******************************************************************************
# demo applications

set( CUDA_SM_ARCH "sm_35" )

set( CMAKE_POSITION_INDEPENDENT_CODE ON )
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CUDA_STANDARD 11 )
set( CMAKE_CUDA_STANDARD_REQUIRED ON )

include_directories( ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------
# demo01: particles stored in global memory, lattice stored in global memory

set( COMMON_HEADERS definitions.h beam_elements.h beamfields.h
                    lattice.h particle.h )

set( CUDA_NVCC_FLAGS "-arch=${CUDA_SM_ARCH};-fmad=false;-lineinfo;-D_FORCE_INLINES" )

CUDA_ADD_EXECUTABLE( demo01_sc0 demo01.cu ${COMMON_HEADERS} )
target_link_libraries( demo01_sc0 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

CUDA_ADD_EXECUTABLE( demo02_sc0 demo02.cu ${COMMON_HEADERS} )
target_link_libraries( demo02_sc0 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

# ------------------------------------------------------------------------------
# demo02: particles stored in global memory, lattice stored in global memory
#         BUT: tracking with thread-local / private copy of the particles,
#              i.e. avoiding the frequent global memory accesses

set( CUDA_NVCC_FLAGS "-arch=${CUDA_SM_ARCH};-fmad=false;-lineinfo;-D_FORCE_INLINES" )
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DDEMOTRACK_ENABLE_BEAMFIELDS" )

CUDA_ADD_EXECUTABLE( demo01_sc1 demo01.cu )
target_link_libraries( demo01_sc1 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )

CUDA_ADD_EXECUTABLE( demo02_sc1 demo02.cu )
target_link_libraries( demo02_sc1 PRIVATE ${DEMOTRACK_LINK_LIBRARIES} )


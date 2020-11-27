# demotrack_cuda
Specialized C++/CUDA version of demotrack. 

## Overview
Implements a minimal particle tracking software for beam-dynamics studies, similar to what [sixtracklib](https://github.com/sixtrack/sixtracklib) or 
[simpletrack](https://github.com/rdemaria/simpletrack) provide. It only implements the bare minimum of necessary beam-elements and infrastructure and is **not** suitable for production use!

Requirements:
- CUDA 8.x or newer (C++11 as kernel language is required)
- [cmake](https://cmake.org) >= 3.11 
- gcc or clang suitable for C++11 enabled CUDA

## Build instructions
Normally, cloning this repository, creating a build directory, running `cmake` from the build directory and then make should be sufficient to build the demos:
```
git clone https://github.com/martinschwinzerl/demotrack_cuda.git
cd demotrack_cuda
mkdir build
cd build
cmake ..
make 
``` 

## List of demo / example files

- `demo01.cu`: Implements a naive implementation of tracking (i.e. particles stored in global memory, lattice stored in global memory, straight-forward implementation of the tracking logic). The tracking kernel is inside `demo01.cu`. It uses a minimal FODO lattice with only 8 elements in the following arrangement:
    - A `Multipole` in a dipole configuration `dipole0`
    - A `Drift` `drift0` with a length of 5m
    - A `Multipole` in a focusing quadrupole configuration `q0`
    - A `Drift` `drift1` with a length of 5m 
    - A `Multipole` in a dipole configuration `dipole1`
    - A `Drift` `drift2` with a length of 5m 
    - A `Multipole` in a de-focusing quadrupole configuration `q1`
    - A `Drift` `drift3` with a length of 5m 
    - A `Cavity` `cavity0` with a voltuage of 5000000V, a frequency of 239833966 Hz and a lag of 180 deg
Please cf. the file `fodo_lattice.h` for details. The particle beam is consists of protons with a kinetic energy of 470 GeV. 
Two executables are generated from `demo01.cu`: 
    - `demo01_sc0`: space-charge / beam-fields are disabled
    - `demo02_sc1`: space-charge / beam-fields are enabled
    
- `demo02.cu`: Similar to `demo01.cu` but now a thread-local copy of the particle data is used for tracking over all turns rather than always hitting the global memory. 

## List of infrastructure / library files
- `CMakeLists.txt` build configuration and setup file. Edit to enable different compiler flags, architectures, etc.
- `definitions.h` typedefs and definitions to abstract away different platforms
- `particle.h` provides the particle model and auxiliary functions 
- `beam_elements.h` provides models for the beam-elements, e.g. `Drift`, `Multipole`, `Cavity`
- `beamfields.h` implementation of space charge / beam field elements like `SpaceChargeCoasting`
- `fodo_lattice.h` setup the simple lattice


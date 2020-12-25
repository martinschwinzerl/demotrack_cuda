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
## Usage Examples
A) Simple FODO lattice
To run `demo01_sc0` with 102400 particles for 100 turns on the default simplified FODO lattice and with default initial conditions for the particles, use
```
cd build
./demo01_sc0 102400 100 
``` 
Note: the same signature applies to the other `demox_scy` applications unless otherwise noted
B) LHC lattice with no beam-beam interaction but with imperfections
Same number of particles and turns as with the previous exampleas, but now both the particles and the lattice are loaded from prepared binary files provided in the `data` subdirectory of this repository:
```
cd build
./demo01_sc0 102400 100 ../data/lhc_no_bb_particles.bin ../data/lhc_no_bb_lattice.bin
``` 
Note: Again, the same signature applies to the other `demox_scy` applications unless otherwise noted

C) Creating a binary file with initial particle state
If you want to run a specific lattice with a different initial particle distribution, `tools/create_particle_data` could be useful. It allows to generate a set of `NUM_PARTICLES` and allows to linearly / uniformly distribute the `x`, `y`, `px`, `py`, `zeta`, and `delta`attributes:

```
cd build/tools
./create_particle_data 

Usage: create_particle_data PATH_TO_PARTICLE_DATA_DUMP NUM_PARTICLES P0C MASS0 [CHARGE0]
            [MIN_X]    [MAX_X]    [MIN_Y]     [MAX_Y]
            [MIN_PX]   [MAX_PX]   [MIN_PY]    [MAX_PY]
            [MIN_ZETA] [MAX_ZETA] [MIN_DELTA] [MAX_DELTA]
``` 

The output path `PATH_TO_PARTICLE_DATA`, the number of particles `NUM_PARTICLES`, the kinetic energy and rest mass of the reference particle `P0C` and `MASS0` respectively are mandatory parameters (both given in units of eV). The reference particle charge `CHARGE0` defaults to a proton, i.e. `1 `and is expressed as multiples of the elementary charge. 

All other lengths (i.e. `x`, `y`, `zeta`) are given metres and default to `0`. 
The relative transversal momenta (`px`, `py`) are given in radians and default to `0`.
The relative momenta deviation `delta` is a unit-less ratio and defaults to `0`.

All command line parameters to `create_particle_data` are positional, i.e. in order to only change one of the later sets of parameters, one has to provide values for all the previous ones as well. `MIN_*` / `MAX_*` parameters have to given pairwise, e.g. providing only one is not sufficient. 

In order to create an initial particle distribution of 1024 protons at 6.5 TeV and a spread in `zeta` from `0.0` to `1e-3`, one would run the tool as follows:
``` 
cd build
tools/create_particle_data ../data/lhc_particles.bin 1024 6.5e12 938.272081e6 1.0 \
    0.0 0.0 0.0 0.0 \
    0.0 0.0 0.0 0.0 \
    0.0 1e-3 0.0 0.0
``` 
**TODO**: Provide a python tool with a more convenient interface / calling convention!

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
    - A `Cavity` `cavity0` with a voltage of 5000000V, a frequency of 239833966 Hz and a lag of 180 deg
        
    Please cf. the file `lattice.h` for details. 
    The default particle beam for the FODO lattice consists of protons with a kinetic energy of 470 GeV. 
    Two executables are generated from `demo01.cu`: 
    - `demo01_sc0`: space-charge / beam-fields are disabled
    - `demo01_sc1`: space-charge / beam-fields are enabled
    
- `demo02.cu`: Similar to `demo01.cu` but now a thread-local copy of the particle data is used for tracking over all turns rather than always hitting the global memory. 

## List of infrastructure / library files
- `CMakeLists.txt` build configuration and setup file. Edit to enable different compiler flags, architectures, etc.
- `definitions.h` typedefs and definitions to abstract away different platforms
- `config.h`generated header file controlling the calculation of the grid dimensions or using a predefined value
- `particle.h` provides the particle model and auxiliary functions 
- `beam_elements.h` provides models for the beam-elements, e.g. `Drift`, `DriftExact`, `Multipole`, `Cavity`, `XYShift`, `SRotation`
- `beamfields.h` implementation of space charge / beam field elements like `SpaceChargeCoasting`
- `lattice.h` setup the simple lattice, infrastructure for loading pre-assembled lattices from binary files

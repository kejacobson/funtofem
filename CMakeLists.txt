cmake_minimum_required(VERSION 3.11)

# required for IN_LIST on "old" versions of cmake
cmake_policy(SET CMP0057 NEW)

# allow user to set {PACKAGE}_ROOT env var to find dependencies
cmake_policy(SET CMP0074 NEW)

project(funtofem
        VERSION 1.0)


option(BUILD_PYTHON_EXT "Build the Python Extension" ON)
option(BUILD_EXAMPLES "Build the examples" ON)

# require strict conformance to C++ standard by default (cmake default is the opposite)
set(CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD 11)

set(BUILD_SHARED_LIBS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(LAPACK REQUIRED)

# turn off the c++ mpi bindings
set(MPI_CXX_SKIP_MPICXX ON)
# really make sure to turn them off for MPT
add_definitions(-DMPI_NO_CPPBIND)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})


include_directories(include)
add_subdirectory(include)
add_subdirectory(src)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if (BUILD_PYTHON_EXT)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(PythonExtensions)
  find_package(Cython REQUIRED)
  find_package(NumPy)
  add_subdirectory(funtofem)
endif(BUILD_PYTHON_EXT)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)

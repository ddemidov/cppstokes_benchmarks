cmake_minimum_required(VERSION 3.7)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()
project(cppstokes)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#--- Boost ------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static versions of Boost libraries" OFF)
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif ()
find_package(Boost REQUIRED COMPONENTS program_options)
cmake_policy(SET CMP0004 OLD)

# --- METIS ----------------------------------------------------------------
find_package(Metis)

# --- AMGCL ----------------------------------------------------------------
find_package(amgcl)
get_property(AMGCL_INCLUDE_DIRS TARGET amgcl::amgcl PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

add_executable(bin2mm bin2mm.cpp)
target_link_libraries(bin2mm amgcl::amgcl Boost::program_options)

add_executable(partition partition.cpp)
target_link_libraries(partition amgcl::amgcl Boost::program_options Metis::metis)

add_executable(amgcl_spc_pre amgcl_spc_pre.cpp)
target_link_libraries(amgcl_spc_pre amgcl::amgcl Boost::program_options)
target_compile_definitions(amgcl_spc_pre PRIVATE PRECOND_SCALAR=double)

add_executable(amgcl_spc_pre_block amgcl_spc_pre.cpp)
target_link_libraries(amgcl_spc_pre_block amgcl::amgcl Boost::program_options)
target_compile_definitions(amgcl_spc_pre_block PRIVATE PRECOND_SCALAR=double BLOCK_U)

add_executable(amgcl_spc_pre_block_mixed amgcl_spc_pre.cpp)
target_link_libraries(amgcl_spc_pre_block_mixed amgcl::amgcl Boost::program_options)
target_compile_definitions(amgcl_spc_pre_block_mixed PRIVATE PRECOND_SCALAR=float BLOCK_U)

add_executable(amgcl_v1 amgcl_v1.cpp)
target_link_libraries(amgcl_v1 amgcl::amgcl Boost::program_options)

# --- MPI ------------------------------------------------------------------
find_package(MPI)
add_library(mpi_target INTERFACE)
target_compile_options(mpi_target INTERFACE ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(mpi_target INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(mpi_target INTERFACE "${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES}")

# --- PETSC ----------------------------------------------------------------
add_library(petsc_target INTERFACE)
target_include_directories(petsc_target INTERFACE $ENV{PETSC_DIR}/include)
find_library(PETSC_LIBRARY petsc PATHS ENV PETSC_DIR PATH_SUFFIXES lib)
target_link_libraries(petsc_target INTERFACE ${PETSC_LIBRARY})

add_executable(petsc_fs petsc_fs.cpp petsc_read.cpp)
target_include_directories(petsc_fs PRIVATE ${AMGCL_INCLUDE_DIRS})
target_link_libraries(petsc_fs petsc_target mpi_target)

add_executable(petsc_v1 petsc_v1.cpp petsc_read.cpp)
target_include_directories(petsc_v1 PRIVATE ${AMGCL_INCLUDE_DIRS})
target_link_libraries(petsc_v1 petsc_target mpi_target)

# --- Pardiso --------------------------------------------------------------
find_package(OpenMP)
find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
if (EIGEN_INCLUDE)

    find_path(MKL_INCLUDE mkl_pardiso.h PATHS $ENV{MKL_ROOT} /opt/intel/mkl PATH_SUFFIXES include)
    find_library(MKL_CORE NAMES mkl_core PATHS $ENV{MKL_ROOT} /opt/intel/mkl PATH_SUFFIXES lib/intel64)
    find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 PATHS $ENV{MKL_ROOT} /opt/intel/mkl PATH_SUFFIXES lib/intel64)
    find_library(MKL_GNU_THREAD NAMES mkl_gnu_thread PATHS $ENV{MKL_ROOT} /opt/intel/mkl PATH_SUFFIXES lib/intel64)
    message(STATUS "MKL: ${MKL_CORE} ${MKL_INTEL_LP64}")

    if (MKL_INCLUDE)
        add_executable(pardiso pardiso.cpp)
        target_include_directories(pardiso PRIVATE ${EIGEN_INCLUDE} ${MKL_INCLUDE})
        target_compile_definitions(pardiso PRIVATE EIGEN_USE_MKL_ALL)
        target_compile_options(pardiso PRIVATE )
        target_link_libraries(pardiso
            OpenMP::OpenMP_CXX
            Boost::program_options
            ${MKL_INTEL_LP64}
            ${MKL_GNU_THREAD}
            ${MKL_CORE}
            )
    endif()
endif()

cmake_minimum_required(VERSION 3.7)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()
project(cppstokes)

#--- Boost ------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static versions of Boost libraries" OFF)
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif ()
find_package(Boost REQUIRED COMPONENTS program_options)

# --- AMGCL ----------------------------------------------------------------
find_package(amgcl)
get_property(AMGCL_INCLUDE_DIRS TARGET amgcl::amgcl PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

add_executable(amgcl_spc amgcl_spc.cpp)
target_link_libraries(amgcl_spc amgcl::amgcl Boost::program_options)
target_compile_definitions(amgcl_spc PRIVATE PRECOND_SCALAR=double)

add_executable(amgcl_spc_mixed amgcl_spc.cpp)
target_link_libraries(amgcl_spc_mixed amgcl::amgcl Boost::program_options)
target_compile_definitions(amgcl_spc_mixed PRIVATE PRECOND_SCALAR=float)

add_executable(amgcl_spc_mixed_block amgcl_spc.cpp)
target_link_libraries(amgcl_spc_mixed_block amgcl::amgcl Boost::program_options)
target_compile_definitions(amgcl_spc_mixed_block PRIVATE PRECOND_SCALAR=float BLOCK_U)

add_executable(amgcl_v1 amgcl_v1.cpp)
target_link_libraries(amgcl_v1 amgcl::amgcl Boost::program_options)

# --- MPI ------------------------------------------------------------------
find_package(MPI)
add_library(mpi_target INTERFACE)
target_compile_options(mpi_target INTERFACE ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(mpi_target INTERFACE ${MPI_CXX_INCLUDE_PATH})
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
string(STRIP "${MPI_CXX_LIBRARIES}" MPI_CXX_LIBRARIES)
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

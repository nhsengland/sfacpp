cmake_minimum_required(VERSION 3.16)
project(pysfacpp_internal VERSION 0.0.1)
option(USE_HEADER_ONLY_DLIB "Use Dlib as header-only library (don't compile cpp files)" ON)
# ---- global settings ----
add_compile_definitions(ARMA_64BIT_WORD)
# set to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 0)
# compiler flags
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_options(-g -O1 -fno-omit-frame-pointer -DARMA_64BIT_WORD) #-fsanitize=address)
else()
    # Production flags (Release)
    add_compile_options(-O3 -DNDEBUG -DARMA_64BIT_WORD) 
    # Optional: Architecture specific optimizations (makes it faster but less portable)
    # add_compile_options(-march=native)
endif()
# force dependencies to be built as static libraries
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# ensure compatability between static libs and python module
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# stop static OpenBLAS leaking symbols that might clash with Numpy
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# 
include(FetchContent)

# =========================================================
# 1. CENTRALIZED BLAS/LAPACK SETUP
# =========================================================
# create an INTERFACE target 'common_blas' that holds the linking info.
# Both Dlib and Armadillo will link to this.
add_library(common_blas INTERFACE)

set(USING_SYSTEM_BLAS OFF)
set(USING_INTERNAL_OPENBLAS OFF)
set(BLAS_CONFIGURED_SUCCESSFULLY OFF)

# --- Priority 1: Check Static OpenBLAS (CRITICAL FOR MAC) ---
if(APPLE)
    # message(STATUS "Math Backend: macOS detected. Forcing clean OpenBLAS build to ensure portability.")
    # set(BLAS_CONFIGURED_SUCCESSFULLY OFF)
    message(STATUS "Math Backend: macOS detected. Using System Accelerate Framework.")
    # Link directly to the Accelerate framework
    target_link_libraries(common_blas INTERFACE "-framework Accelerate")
    set(USING_SYSTEM_BLAS ON)
    set(BLAS_CONFIGURED_SUCCESSFULLY ON)
# --- Priority 2: Check Standard System BLAS (LINUX/WINDOWS) ---
else()
    # On Linux, we prefer standard PkgConfig or FindBLAS
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(OpenBLAS openblas)
    endif()

    if(OpenBLAS_FOUND)
        message(STATUS "Math Backend: Found System OpenBLAS (PkgConfig).")
        target_link_libraries(common_blas INTERFACE ${OpenBLAS_LIBRARIES})
        target_include_directories(common_blas INTERFACE ${OpenBLAS_INCLUDE_DIRS})
        set(USING_SYSTEM_BLAS ON)
        set(BLAS_CONFIGURED_SUCCESSFULLY ON)
    else()
        # Fallback to generic FindBLAS (e.g. MKL, Atlas, or distro OpenBLAS)
        find_package(BLAS QUIET)
        find_package(LAPACK QUIET)

        if(BLAS_FOUND AND LAPACK_FOUND)
            message(STATUS "Math Backend: Found System BLAS/LAPACK.")
            target_link_libraries(common_blas INTERFACE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            set(USING_SYSTEM_BLAS ON)
            set(BLAS_CONFIGURED_SUCCESSFULLY ON)
        endif()
    endif()
endif()

# --- Priority 3: Fallback - Compile OpenBLAS from Source (Universal) ---
if(NOT BLAS_CONFIGURED_SUCCESSFULLY)
    message(STATUS "Math Backend: No suitable binary BLAS found. Compiling OpenBLAS from source...")

    FetchContent_Declare(
        openblas
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS
        GIT_TAG        v0.3.30
    )

    # OpenBLAS Build Options
    set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) # Always build static internally
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    # 1. Disable Threading (Removes _GOMP_ and _pthread_ dependencies)
    set(USE_THREAD OFF CACHE BOOL "" FORCE)
    set(USE_OPENMP OFF CACHE BOOL "" FORCE)
    set(NUM_THREADS 1 CACHE STRING "" FORCE)
    
    # On Mac Source builds, ensure we hide symbols too
    if(APPLE)
        set(CMAKE_C_VISIBILITY_PRESET hidden)
    endif()

    # Fortran Check
    enable_language(Fortran OPTIONAL)
    if(NOT CMAKE_Fortran_COMPILER)
        set(NO_FORTRAN ON CACHE BOOL "" FORCE)
        message(WARNING "No Fortran compiler. OpenBLAS will be slower (C-only).")
    else()
        set(NO_FORTRAN OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(openblas)

    target_link_libraries(common_blas INTERFACE openblas)
    find_package(Threads REQUIRED)
    target_link_libraries(common_blas INTERFACE Threads::Threads)
    
    set(USING_INTERNAL_OPENBLAS ON)
endif()

# ---- preconfigure boost ----
set(BOOST_INCLUDE_LIBRARIES math optional)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  Boost
  URL https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.gz
  URL_HASH SHA256=5e93d582aff26868d581a52ae78c7d8edf3f3064742c6e77901a1f18a437eea9
#   GIT_REPOSITORY https://github.com/boostorg/boost.git
#   GIT_TAG boost-1.89.0
)
# ---- preconfigure dlib ----
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "Disable GUI support for Python" FORCE)
set(DLIB_USE_BLAS ON CACHE BOOL "Use BLAS" FORCE) 
# set(DLIB_USE_LAPACK ON CACHE BOOL "Use LAPACK" FORCE)
# disable dlib's attempt to link system LAPACK so it doesn't clash with Armadillo.
set(DLIB_USE_LAPACK OFF CACHE BOOL "Disable LAPACK to prevent conflict" FORCE)
set(DLIB_USE_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
set(DLIB_PNG_SUPPORT OFF CACHE BOOL "Diasable PNG support" FORCE)
set(DLIB_JPEG_SUPPORT OFF CACHE BOOL "Disable JPEG support" FORCE)
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v20.0
)
# ---- armadillo ----
FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG 15.2.2
    GIT_SHALLOW TRUE
)
# ---- preconfigure spdlog ----
set(SPDLOG_BUILD_EXAMPLE OFF CACHE INTERNAL "")
set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
set(SPDLOG_BUILD_SHARED OFF CACHE INTERNAL "")
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.16.0
)
# ---- pybind11 ----
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.1
)
# ---- carma ---
# FetchContent_Declare(
#     carma
#     GIT_REPOSITORY https://github.com/RUrlus/carma.git
#     GIT_TAG        v0.8.0
# )
# ---- thread pool ----
FetchContent_Declare(
    bshoshany_thread_pool
    URL https://github.com/bshoshany/thread-pool/archive/refs/tags/v5.0.0.tar.gz
)
# fetch everything except armadillo
FetchContent_MakeAvailable(Boost pybind11 spdlog bshoshany_thread_pool)
# ---- conditional handling of dlib ----
FetchContent_GetProperties(dlib)
if (NOT dlib_POPULATED)
    if (USE_HEADER_ONLY_DLIB)
        # header only - dont compile
        message(STATUS "Dlib strategy: HEADER-ONLY {no compilation}")
        FetchContent_Populate(dlib)
        # create the target (since didn't run the CMake)
        if (NOT TARGET dlib::dlib)
            add_library(dlib::dlib INTERFACE IMPORTED)
            target_include_directories(dlib::dlib INTERFACE "${dlib_SOURCE_DIR}")
        endif()
    else()
        # compile from source
        message(STATUS "Dlib strategy: BUILD FROM SRC")
        FetchContent_MakeAvailable(dlib)
    endif()
endif()
# ---- handle armadillo ----
# manually populate Armadillo (Download headers, but DO NOT run its CMake)
# Allow manual population (Suppress CMP0169 error)
# if(POLICY CMP0169)
#   cmake_policy(SET CMP0169 OLD)
# endif()
FetchContent_GetProperties(armadillo)
if(NOT armadillo_POPULATED)
    FetchContent_Populate(armadillo)
endif()
# ---- Configure Armadillo Target ----
# include_directories(BEFORE SYSTEM "${armadillo_SOURCE_DIR}/include")
message(STATUS "Force-Injecting Armadillo Headers: ${armadillo_SOURCE_DIR}/include")
set(CMAKE_CXX_FLAGS "-I\"${armadillo_SOURCE_DIR}/include\" ${CMAKE_CXX_FLAGS}")

if(NOT TARGET Armadillo::Armadillo)
    add_library(Armadillo::Armadillo INTERFACE IMPORTED)
    target_include_directories(Armadillo::Armadillo INTERFACE "${armadillo_SOURCE_DIR}/include")
endif()
# Apply Armadillo settings
# apparently openblas has some overhead for small matricies?
# Bypass BLAS for matrices smaller than 10x10
# ARMA_MIN_BLAS_SIZE=100
target_compile_definitions(
    Armadillo::Armadillo INTERFACE 
        ARMA_DONT_USE_WRAPPER 
        ARMA_NO_DEBUG
        ARMA_DONT_PRINT_ERRORS
)
# Link Armadillo to our Centralized Math Backend
target_link_libraries(Armadillo::Armadillo INTERFACE common_blas)
# ---- Configure Dlib Target ----
if(USING_INTERNAL_OPENBLAS)
    # Use PUBLIC so the BLAS dependency propagates to anything that uses dlib
    if(TARGET dlib)
        target_link_libraries(dlib PUBLIC common_blas)
    endif()
endif()

# ---- threadpool ----
add_library(bs_thread_pool INTERFACE)
target_include_directories(bs_thread_pool INTERFACE ${bshoshany_thread_pool_SOURCE_DIR}/include)

# Find Python and Pybind11
# set(Python_FIND_VIRTUALENV ONLY)
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# parent directory
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
# all source files
file (GLOB SFILES "${PARENT_DIR}/src/*.cpp")
# remove RcppExports from python build
list(REMOVE_ITEM SFILES "${PARENT_DIR}/src/RcppExports.cpp")
list(REMOVE_ITEM SFILES "${PARENT_DIR}/src/main.cpp")
# **** INCLUDES ****
# header files for package
include_directories(pysfacpp_internal PUBLIC "${PARENT_DIR}/include")
include_directories(SYSTEM "${boost_SOURCE_DIR}")
if (EXISTS "${spdlog_SOURCE_DIR}/include")
    include_directories(SYSTEM "${spdlog_SOURCE_DIR}/include")
else()
    message(WARNING "spdlog source directiory not found at ${spdlog_SOURCE_DIR}")
endif()

pybind11_add_module(pysfacpp_internal ${SFILES})

# linkage
target_link_libraries(pysfacpp_internal 
    PRIVATE
        Armadillo::Armadillo
        dlib::dlib
        # Boost::math
        # Boost::optional
        # boost_math
        # boost_optional
        spdlog::spdlog_header_only
        bs_thread_pool
)
# target_compile_definitions(pysfacpp_internal PRIVATE ARMA_DONT_USE_BLAS)
target_compile_definitions(
    pysfacpp_internal
    PRIVATE
        WITHDLIB # using dlib for optimization
        PYPACKAGE # compiling python package
        VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
if (USE_HEADER_ONLY_DLIB)
    target_compile_definitions(pysfacpp_internal PRIVATE DLIB_THREAD_POOL_TIMEOUT=0)
endif()
# OpenBLAS should use single-threaded
target_compile_definitions(pysfacpp_internal PRIVATE OPENBLAS_NUM_THREADS=1)
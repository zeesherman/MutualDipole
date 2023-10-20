project(MutualDipole)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

include(FindHOOMD.cmake)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Installation prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install plugin to: " ${PYTHON_MODULE_BASE_DIR})

# We use AmgX's implementation of CUSP. Find this.
find_path(AMGX_INCLUDE_DIRECTORY cusp/version.h)
if( NOT AMGX_INCLUDE_DIRECTORY )
         message(FATAL_ERROR "cusp not found")
endif( NOT AMGX_INCLUDE_DIRECTORY )
message(STATUS  "Found cusp: ${AMGX_INCLUDE_DIRECTORY}")

# Add AmgX's include directory.
include_directories(BEFORE ${AMGX_INCLUDE_DIRECTORY})

# add subdirectories
add_subdirectory(${PROJECT_NAME})

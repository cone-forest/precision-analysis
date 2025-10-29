# Install script for directory: C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/pybind11_example")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/AdolcForward"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/AlignedVector3"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/ArpackSupport"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/AutoDiff"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/BVH"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/EulerAngles"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/FFT"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/IterativeSolvers"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/KroneckerProduct"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/LevenbergMarquardt"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/MatrixFunctions"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/MPRealSupport"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/NNLS"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/NonLinearOptimization"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/NumericalDiff"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/OpenGLSupport"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/Polynomials"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/SparseExtra"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/SpecialFunctions"
    "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/Users/User/Desktop/precision-analysis/extern/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/User/Desktop/precision-analysis/build/extern/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/User/Desktop/precision-analysis/build/extern/eigen/unsupported/Eigen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()

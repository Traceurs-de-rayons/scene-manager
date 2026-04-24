cmake_minimum_required(VERSION 3.16)
project(scene-manager VERSION 0.1.0 LANGUAGES CXX)

# Create an INTERFACE library.
# This is the correct type for a header-only library or a library with no source files.
# It doesn't produce a compiled artifact but can be used to specify usage requirements
# for other targets that link against it.
add_library(scene-manager INTERFACE)

# Specify the include directories that consumers of this library will need.
# The INTERFACE keyword ensures that any target linking to scene-manager
# will automatically have this directory added to its include path.
target_include_directories(scene-manager
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Propagate C++ standard requirements to consumers of this library.
target_compile_features(scene-manager INTERFACE cxx_std_23)

# Define dependencies for this interface library.
# Any target linking against scene-manager will also be linked against these libraries.
# I'm assuming a dependency on scene-core as it seems logical.
if(NOT TARGET scene-core)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../scene-core scene-core-build)
endif()

# Add dependency on raster-core
if(NOT TARGET raster-core)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../raster-core raster-core-build)
endif()

target_link_libraries(scene-manager INTERFACE scene-core raster-core)
